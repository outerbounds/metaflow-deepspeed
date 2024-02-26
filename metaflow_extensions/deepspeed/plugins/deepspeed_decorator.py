from metaflow.unbounded_foreach import UBF_CONTROL
from metaflow.plugins.parallel_decorator import (
    ParallelDecorator,
    _local_multinode_control_task_step_func,
)
from metaflow.exception import MetaflowException
import metaflow

# from metaflow import Run as MetaflowRun
from functools import partial
from typing import List, Dict, Union
import subprocess
import socket
import json
import time
import sys
import os
from io import BytesIO
from collections import namedtuple


HOSTFILE_IP_KEY = "hostfile_ips"
HOSTFILE = "hostfile.txt"
DEEPSPEED_JOB_COMPLETE_VAR = "mpi_job_status"
PUBLIC_KEY_RECEIVED_VAR = "public_key_received"
CONTROL_TASK_DONE_PATH = "control"
MPI_PUBLIC_KEY_PATH = "public_keys"
DEEPSPEED_ENV_FILE = ".deepspeed_env"  # https://github.com/microsoft/DeepSpeed/blob/24f20ef0a105d32f6085fe0d3b1c2f9324a6262c/docs/_tutorials/getting-started.md?plain=1#L230-L254
DEEPSPEED_SUFFIX = "deepspeed_datastore"


class DeepspeedExecutor:
    """
    Instances of the DeepspeedExecutor class are responsible for launching the Deepspeed command. There is one per Metaflow @step annotated with @deepspeed.
    DeepspeedExecutor consumes a list of hosts aka nodes aka k8s pods aka metaflow task containers, and information about how many processes to run on each.
    In the constructor, the _scan_all_hosts function constructs the ~/.ssh/known_hosts list on each node, so they know how to communicate via passwordless ssh.

    The Deepspeed decorator, which users specify in a Metaflow num_parallel task with @deepspeed, attaches an instance of this class to current.deepspeed.
        Using current.deepspeed.run, users can run the same Deepspeed launch command they would independently of Metaflow.
        This class will handle opening the subprocess, and ensuring other typical Metaflow functionality works as expected.
    """

    def __init__(
        self,
        hosts: List[str],
        n_slots_per_host: int = 1,
        is_gpu: bool = False,
        flow=None,
        worker_polling_freq: int = 10,
        flow_datastore=None,
    ) -> None:
        self.is_gpu = is_gpu
        self.n_slots_per_host = n_slots_per_host
        self.hosts = [
            h for h in hosts if h != socket.gethostbyname(socket.gethostname())
        ] + [
            "127.0.0.1"
        ]  # control node can use localhost
        self._scan_all_hosts()
        self.flow = flow
        self.worker_polling_freq = worker_polling_freq
        self._flow_datastore = flow_datastore

    def _exec_cmd(
        self,
        deepspeed_args: Union[List[str], Dict[str, str]] = [],
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = [],
    ):
        """
        deepspeed_args: Dict[str] - arguments to pass to `exe`
        entrypoint: str - Python script for the Deepspeed launcher to run, such as a PyTorch or Huggingface training routine.
        entrypoint_args: Dict[str] - arguments to pass after `entrypoint`
        """

        # Container to build up the command to be run in a subprocess.
        cmd = ["deepspeed"]

        # Construct the deepspeed distributed args.
        if "--hostfile" not in deepspeed_args:
            deepspeed_args.extend(["--hostfile", HOSTFILE])
        if "--num_nodes" not in deepspeed_args:
            deepspeed_args.extend(["--num_nodes", str(len(self.hosts))])
        if self.is_gpu and "--num_gpus" not in deepspeed_args:
            deepspeed_args.extend(["--num_gpus", str(self.n_slots_per_host)])
        if "--master_addr" in deepspeed_args:
            raise MetaflowException(
                "Do not specify the --master_addr in your current.run.deepspeed args. Metaflow will set this for you."
            )
        my_ip = socket.gethostbyname(socket.gethostname())
        deepspeed_args.extend(["--master_addr", my_ip])
        cmd.extend(deepspeed_args)

        # Construct rest of command starting with the entrypoint.
        if entrypoint is not None:
            cmd.append(entrypoint)
        else:
            raise MetaflowException(
                "current.deepspeed.run(..., entrypoint=<SCRIPT>, ...) arg must be specified."
            )
        # cmd.extend(entrypoint_args)
        if entrypoint_args is not None and isinstance(entrypoint_args, dict):
            for arg, val in entrypoint_args.items():
                if val == "" or val == None:
                    cmd.append("--%s" % arg)
                else:
                    cmd.extend(["--%s" % arg, str(val)])
        elif entrypoint_args is not None and isinstance(entrypoint_args, list):
            cmd.extend(entrypoint_args)

        # Write env_var=value to file.
        # Deepspeed automatically looks for this file to prepend the variables to its launcher command.
        with open(DEEPSPEED_ENV_FILE, "w") as f:
            for k, v in os.environ.items():
                if k == "METAFLOW_INIT_SCRIPT":
                    continue  # Don't pass these to deepspeed. Because of how deepspeed reads env vars and sets them in runner commands.
                else:
                    json_string = json.dumps(v)
                    f.write(f"{k}={json_string}\n")

        # Launch the Deepspeed run.
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,  # pipe to Metaflow stdout. TODO: How to handle progress bar buffering like TQDM.
            stderr=subprocess.STDOUT,
        ) as process:
            while process.poll() is None:
                stdout = process.stdout.read1()
                try:
                    text = stdout.decode("utf-8")
                except UnicodeDecodeError:
                    text = ""
                print(
                    text, end="", flush=True
                )  # TODO: other mechanism to flush buffer, especially in progress bar case.

            if process.returncode != 0:
                raise DeepspeedException(cmd)

    def run(
        self,
        deepspeed_args: Union[List[str], Dict[str, str]] = [],
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = [],
        push_results_dir_to_s3: bool = False,
        local_output_dir: str = None,
        s3_output_dir: str = None,
        s3_root: str = None,
    ) -> str:
        from metaflow import current

        node_index = current.parallel.node_index  # assumes parallel
        datastore = DeepspeedDatastore(
            flow_datastore=self._flow_datastore, pathspec=current.pathspec
        )

        if push_results_dir_to_s3 and datastore._backend.TYPE != "s3":
            raise MetaflowException(
                "current.deepspeed.run must use S3 as a datastore if push_results_dir_to_s3 is True"
            )
        elif push_results_dir_to_s3:
            # TODO: Annoying place for this check. Consider moving the S3 push args into the decorator itself, so can be checked at flow init instead.
            if local_output_dir is None:
                raise MetaflowException(
                    "current.deepspeed.run must specify local_output_dir if push_results_dir_to_s3 is True"
                )
            elif s3_output_dir is None:
                if local_output_dir.startswith("/"):
                    s3_output_dir = local_output_dir[1:]
                else:
                    s3_output_dir = local_output_dir

        # Run the distributed job
        if node_index == 0:  # control node
            self._exec_cmd(deepspeed_args, entrypoint, entrypoint_args)
            datastore.put(
                CONTROL_TASK_DONE_PATH, json.dumps({DEEPSPEED_JOB_COMPLETE_VAR: True})
            )
        else:  # worker node
            control_done = False
            while not control_done:
                time.sleep(self.worker_polling_freq)
                try:
                    control_done = json.loads(
                        datastore.get(CONTROL_TASK_DONE_PATH).blob
                    )[DEEPSPEED_JOB_COMPLETE_VAR]
                except DeepspeedDatastoreNotFoundError:
                    control_done = False
                    continue

        # Push results to S3
        if push_results_dir_to_s3 and datastore._backend.TYPE == "s3":
            if s3_root is not None:
                s3 = S3(s3root=s3_root)
            else:
                s3 = S3(run=self.flow)
            if not os.path.exists(local_output_dir):
                print(
                    f"Deepspeed process completed, and local_output_dir `{local_output_dir}` does not exist, skipping push to S3."
                )
                return
            if not os.path.isdir(local_output_dir):
                print(
                    f"Deepspeed process completed, and local_output_dir `{local_output_dir}` is not a directory, skipping push to S3."
                )
                return
            if len(os.listdir(local_output_dir)) == 0:
                print(
                    f"Deepspeed process completed, and local_output_dir `{local_output_dir}` is empty, skipping push to S3."
                )
                return
            filepath_tuples = []
            for path, subdirs, files in os.walk(local_output_dir):
                for fname in files:
                    filepath_tuples.append(
                        (
                            f"{s3_output_dir}/{str(node_index)}/{os.path.relpath(os.path.join(path, fname), local_output_dir)}",
                            os.path.join(path, fname),
                        )
                    )
            print(
                f"Pushing outputs in {local_output_dir} from node {node_index} to S3..."
            )
            path_result = s3.put_files(filepath_tuples)
            s3_output_dir_full = (
                f"{path_result[0][1].split(f'/{s3_output_dir}')[0]}/{s3_output_dir}"
            )
            print(f"Push completed. Results available at {s3_output_dir_full}.")
            print(
                f"\nTo access the S3 results, instantiate Metaflow's S3 client using:\n\twith S3(run=Run('{current.flow_name}/{current.run_id}')) as s3: ..."
            )
            print(
                f"\nTo view metadata from this node use:\n\ts3.list_paths(['{s3_output_dir}/{node_index}'])"
            )
            print(
                f"\nTo recurisvely download everything in {s3_output_dir_full} use:\n\ts3.get_recursive(keys=['{s3_output_dir}'])\n\n"
            )
            s3.close()
            return s3_output_dir_full

    def _scan_cmd(self, host):
        return ["ssh-keyscan", "-H", host]

    def _scan_all_hosts(self):
        for host in self.hosts:
            try:
                result = subprocess.run(
                    self._scan_cmd(host),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                if result.returncode != 0:
                    raise Exception(
                        "Error adding host to known_hosts: "
                        + result.stdout.decode("utf-8")
                    )
                else:
                    with open(os.path.expanduser("~/.ssh/known_hosts"), "a") as f:
                        f.write(result.stdout.decode("utf-8"))
            except subprocess.CalledProcessError as e:
                print(e.stdout)
                raise e


# mimic a subset of the behavior of the Metaflow S3Object
DeepspeedDatastoreBlob = namedtuple("DeepspeedDatastoreBlob", "blob url text")
DeepspeedListPathResult = namedtuple("DeepspeedListPathResult", "url")


class DeepspeedDatastore(object):

    """
    This class is a wrapper around the basic Metaflow cloud datastore functionality.
    It is used to interact with each cloud datastore provider from within the DeepspeedExecutor and DeepspeedDecorator class.
    For now, local storage is not supported.
    Methods provided follow the naming convention of Metaflow's S3 client: put, get, and list_paths.
    """

    def __init__(self, flow_datastore, pathspec=None):
        self._backend = flow_datastore._storage_impl
        self._flow_name = flow_datastore.flow_name
        _, run_id, step_name, _ = pathspec.split("/")
        self._run_id = run_id
        self._step_name = step_name
        self._pathspec = pathspec

    @property
    def get_storage_root(self):
        """
        Return the path to the root of the deepspeed datastore.
        This method is where the unique deepspeed datastore root for each cloud provider is specified.

        Note: S3Storage class uses the S3 client (other clouds do not have this), 
            which prepends the storage root inside the self._backend calls this class uses.
        """
        if self._backend.TYPE == "s3":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

            return DEEPSPEED_SUFFIX
        elif self._backend.TYPE == "azure":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_AZURE

            return os.path.join(DATASTORE_SYSROOT_AZURE, DEEPSPEED_SUFFIX)
        elif self._backend.TYPE == "gs":
            from metaflow.metaflow_config import DATASTORE_SYSROOT_GS

            return os.path.join(DATASTORE_SYSROOT_GS, DEEPSPEED_SUFFIX)
        else:
            raise NotImplementedError(
                "Deepspeed datastore does not support backend %s" % (self._backend.TYPE)
            )

    def get_datastore_file_location(self, key):
        return os.path.join(
            self.get_storage_root, self._flow_name, self._run_id, self._step_name, key
        )

    def put(self, key: str, value: str):
        "Put a single object into the datastore's `key` index."
        self._backend.save_bytes(
            [(self.get_datastore_file_location(key), BytesIO(value.encode("utf-8")))],
            overwrite=True,
        )

    def get(self, key):
        "Get a single object residing in the datastore's `key` index."
        datastore_url = self.get_datastore_file_location(key)
        with self._backend.load_bytes([datastore_url]) as get_results:
            for key, path, meta in get_results:
                if path is not None:
                    with open(path, "rb") as f:
                        blob_bytes = f.read()
                        return DeepspeedDatastoreBlob(
                            blob=blob_bytes,
                            url=datastore_url,
                            text=blob_bytes.decode("utf-8"),
                        )
                else:
                    raise DeepspeedDatastoreNotFoundError(datastore_url)

    def get_many(self, keys):
        return [self.get(key) for key in keys]

    def list_paths(self, keys):
        "List all objects in the datastore's `keys` index."
        if self._backend.TYPE == "gs":
            raise NotImplementedError(
                """Deepspeed datastore does not support the list_paths API for Google Cloud storage. 
                If you know the paths ahead of this call, use get_many with the keys you want to list instead."""
            )
        keys = [self.get_datastore_file_location(key) for key in keys]
        list_path_results = [
            DeepspeedListPathResult(url=list_content_result.path)
            for list_content_result in self._backend.list_content(keys)
        ]
        return list_path_results


class DeepspeedDecorator(ParallelDecorator):

    name = "deepspeed"
    defaults = {"all_nodes_started_timeout": 90, "worker_polling_freq": 5}
    IS_PARALLEL = True

    requires_passwordless_ssh = (
        True  # modifies how @kubernetes runs command in docker container
    )

    def _setup_current(self, hosts, n_slots_per_host, is_gpu, flow):
        from metaflow import current

        current._update_env(
            {
                "deepspeed": DeepspeedExecutor(
                    hosts,
                    n_slots_per_host,
                    is_gpu,
                    flow,
                    self.attributes["worker_polling_freq"],
                    self.flow_datastore,
                )
            }
        )

    def step_init(self, flow, graph, step, decos, environment, flow_datastore, logger):
        from metaflow.plugins.kubernetes.kubernetes_decorator import KubernetesDecorator
        from metaflow.plugins.aws.batch.batch_decorator import BatchDecorator
        from metaflow.plugins.aws.aws_utils import compute_resource_attributes

        self.environment = environment
        self.flow_datastore = flow_datastore

        self.is_batch = False
        self.is_k8s = False
        self.is_local = False
        for deco in decos:
            if isinstance(deco, KubernetesDecorator):
                self.is_k8s = True
                break
            elif isinstance(deco, BatchDecorator):
                self.is_batch = True
                break
        else:
            self.local = True

        for deco in decos:
            if deco.name in ["resources", "kubernetes", "batch"]:
                compute_deco_attrs = compute_resource_attributes(
                    decos, deco, step, {"cpu": "1", "gpu": "0"}
                )
                try:
                    self.n_slots = int(compute_deco_attrs["gpu"])
                    self.is_gpu = True
                except KeyError:
                    self.n_slots = int(compute_deco_attrs["cpu"])
                    self.is_gpu = False
                if not self.n_slots > 0:
                    self.n_slots = int(compute_deco_attrs["cpu"])
                    self.is_gpu = False
                break

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        def _step_func_with_setup():
            self.setup_distributed_env(flow, ubf_context)
            step_func()

        if (
            ubf_context == UBF_CONTROL
            and os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local"
        ):
            from functools import partial

            env_to_use = getattr(self.environment, "base_env", self.environment)

            return partial(
                _local_multinode_control_task_step_func,
                flow,
                env_to_use,
                _step_func_with_setup,
                retry_count,
            )
        else:
            return _step_func_with_setup

    def setup_distributed_env(self, flow, ubf_context):
        "Return a list of strings of hostnames of nodes to use for MPI"
        hosts = setup_mpi_env(
            flow,
            ubf_context,
            self.attributes["all_nodes_started_timeout"],
            self.attributes["worker_polling_freq"],
            self.n_slots,
            self.is_k8s,
            self.flow_datastore,
        )
        self._setup_current(
            hosts=hosts, n_slots_per_host=self.n_slots, is_gpu=self.is_gpu, flow=flow
        )
        return hosts


def setup_mpi_env(
    run,
    ubf_context,
    all_nodes_started_timeout,
    interval,
    n_slots,
    is_k8s,
    flow_datastore,
):
    # NOTE: Where jobset for @kuberentes + @parallel can automate the sshd port opening,
    # AWS Batch will require security groups applied to the compute env for this.

    from metaflow import current

    datastore = DeepspeedDatastore(
        flow_datastore=flow_datastore, pathspec=current.pathspec
    )

    # gather distributed universe variables
    if is_k8s:
        world_size = int(os.environ["WORLD_SIZE"])
        if ubf_context == UBF_CONTROL:
            node_index = 0
        else:
            # node_index = int(os.environ["RANK"]) + 1
            node_index = int(os.environ["RANK"])
    else:  # is_batch
        world_size = int(os.environ["AWS_BATCH_JOB_NUM_NODES"])
        node_index = int(os.environ["AWS_BATCH_JOB_NODE_INDEX"])

    if ubf_context == UBF_CONTROL:
        key_push_path = "%s/control" % (MPI_PUBLIC_KEY_PATH)
    else:
        key_push_path = "%s/worker/%s" % (MPI_PUBLIC_KEY_PATH, node_index)

    my_ip = socket.gethostbyname(socket.gethostname())
    ssh_dir = os.path.expanduser("~/.ssh")
    if not os.path.exists(ssh_dir):
        os.makedirs(ssh_dir)
    os.chmod(ssh_dir, 0o700)

    # Generate host key
    result = subprocess.run(
        ["ssh-keygen", "-t", "rsa", "-f", os.path.join(ssh_dir, "id_rsa"), "-N", ""],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert result.returncode == 0, "Error generating host key"
    # Move public key to S3
    with open(os.path.join(ssh_dir, "id_rsa.pub"), "r") as f:
        datastore.put(key_push_path, f.read())
    if ubf_context == UBF_CONTROL:
        # loop until all workers have pushed their public keys
        worker_keys_path = "%s/worker" % MPI_PUBLIC_KEY_PATH
        while True:
            try:
                paths = []
                num_workers_registered = 0
                for i in range(1, world_size): # worker node indices start at 1
                    path = os.path.join(worker_keys_path, str(i))
                    try:
                        datastore.get(path)
                        num_workers_registered += 1
                        paths.append(path)
                    except DeepspeedDatastoreNotFoundError:
                        pass
                if num_workers_registered == world_size - 1:  # all nodes minus control
                    break
                time.sleep(interval)
            except DeepspeedDatastoreNotFoundError:
                time.sleep(interval)
                continue
        # append all public keys to authorized_keys file
        with open(os.path.join(ssh_dir, "authorized_keys"), "a") as g:
            for p in paths:
                obj = datastore.get(p)
                g.write(obj.text)
            # add self to keys too
            with open(os.path.join(ssh_dir, "id_rsa.pub"), "r") as f:
                g.write(f.read())
    else:
        control_key_path = "%s/control" % MPI_PUBLIC_KEY_PATH
        while True:
            try:
                obj = datastore.get(control_key_path)
                with open(os.path.join(ssh_dir, "authorized_keys"), "a") as g:
                    g.write(obj.text)
                break
            except DeepspeedDatastoreNotFoundError:
                time.sleep(interval)
                continue
    os.chmod(os.path.join(ssh_dir, "authorized_keys"), 0o600)

    # enable passwordless ssh
    ssh_config_options = [
        "PubKeyAuthentication yes",
        "RSAAuthentication yes",
    ]
    with open("/etc/ssh/sshd_config", "a") as f:
        f.write("\n".join(ssh_config_options))
    result = subprocess.run(
        ["sudo", "service", "ssh", "restart"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, "Error restarting sshd"

    # Share IPs for writing the hostfiles
    datastore.put("%s/%s" % (HOSTFILE_IP_KEY, node_index), my_ip)
    while True:
        datastore_hostfile_entry_paths = []
        for i in range(world_size):
            try:
                ip_object_store_key = "%s/%s" % (HOSTFILE_IP_KEY, i)
                _ = datastore.get(ip_object_store_key)
                datastore_hostfile_entry_paths.append(ip_object_store_key)
            except DeepspeedDatastoreNotFoundError:
                pass
        if len(datastore_hostfile_entry_paths) == world_size:
            hosts = []
            for datastore_path in datastore_hostfile_entry_paths:
                hosts.append(
                    datastore.get(datastore_path).blob.decode("utf-8")
                )
            break
        time.sleep(5)

    with open(HOSTFILE, "a") as f:
        for h in hosts:
            if h == my_ip:
                h = "127.0.0.1"
            f.write("%s slots=%s\n" % (h, n_slots))

    return hosts


class DeepspeedException(MetaflowException):
    headline = ""

    def __init__(self, cmd):
        msg = "The Deepspeed command \n\n{}\n\nfailed to complete.".format(
            " ".join(cmd)
        )
        super(DeepspeedException, self).__init__(msg)


class DeepspeedDatastoreNotFoundError(MetaflowException):
    headline = "DeepSpeed Datastore Not Found"

    def __init__(self, datastore_path_name):
        msg = "The DeepSpeed datastore path {} was not found.".format(
            datastore_path_name
        )
        super(DeepspeedDatastoreNotFoundError, self).__init__(msg)
