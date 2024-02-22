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
import io

HOSTFILE_IP_KEY = "hostfile_ips"
HOSTFILE = "hostfile.txt"
DEEPSPEED_JOB_COMPLETE_VAR = "mpi_job_status"
PUBLIC_KEY_RECEIVED_VAR = "public_key_received"
CONTROL_TASK_DONE_PATH = "control"
MPI_PUBLIC_KEY_PATH = "public_keys"
DEEPSPEED_ENV_FILE = ".deepspeed_env"  # https://github.com/microsoft/DeepSpeed/blob/24f20ef0a105d32f6085fe0d3b1c2f9324a6262c/docs/_tutorials/getting-started.md?plain=1#L230-L254


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
    ) -> None:
        self.is_gpu = is_gpu
        self.n_slots_per_host = n_slots_per_host
        self.hosts = [
            h for h in hosts if h != socket.gethostbyname(socket.gethostname())
        ] + ["127.0.0.1"]
        self._scan_all_hosts()
        self.flow = flow
        self.worker_polling_freq = worker_polling_freq

    def _exec_cmd(
        self,
        deepspeed_args: Union[List[str], Dict[str, str]] = [],
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = [],
        # deepspeed_config = None,
    ):
        """
        deepspeed_args: Dict[str] - arguments to pass to `exe`
        entrypoint: str - Python script for the Deepspeed launcher to run, such as a PyTorch or Huggingface training routine.
        entrypoint_args: Dict[str] - arguments to pass after `entrypoint`
        deepspeed_config: ... TODO
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
                    # TODO: investigate PR to deepspeed to change how it uses .deepspeed_env to set env vars on workers.
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
                )  # TODO: explore other mechanisms to flush buffer correctly, especially in progress bar case.

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
        from metaflow import current, S3

        node_index = current.parallel.node_index  # assumes parallel
        if push_results_dir_to_s3:
            # TODO: Annoying place for this check. Consider moving the S3 push args into the decorator itself, so can be checked at flow init instead.
            if local_output_dir is None:
                raise MetaflowException(
                    "current.deepspeed.run must specify local_output_dir if push_results_dir_to_s3 is True"
                )
            elif s3_output_dir is None:
                if local_output_dir.startswith("/"):
                    s3_output_dir = local_output_dir[1:]
                else:
                    s3_output_dir = local_output_dir  # os.path.relpath(local_output_dir, os.getcwd())

        if s3_root is not None:
            s3 = S3(s3root=s3_root)
        else:
            s3 = S3(run=self.flow)

        # Run the distributed job
        if node_index == 0:  # control node
            self._exec_cmd(deepspeed_args, entrypoint, entrypoint_args)
            s3.put(
                CONTROL_TASK_DONE_PATH, json.dumps({DEEPSPEED_JOB_COMPLETE_VAR: True})
            )
        else:  # worker node
            control_done = False
            while not control_done:
                time.sleep(self.worker_polling_freq)
                try:
                    control_done = json.loads(s3.get(CONTROL_TASK_DONE_PATH).blob)[
                        DEEPSPEED_JOB_COMPLETE_VAR
                    ]
                except metaflow.plugins.datatools.s3.s3.MetaflowS3NotFound:
                    control_done = False
                    continue

        # Push results to S3
        if push_results_dir_to_s3:
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

        s3.close()

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
                )
            }
        )

    def step_init(self, flow, graph, step, decos, environment, flow_datastore, logger):
        from metaflow.plugins.kubernetes.kubernetes_decorator import KubernetesDecorator
        from metaflow.plugins.aws.batch.batch_decorator import BatchDecorator
        from metaflow.plugins.aws.aws_utils import compute_resource_attributes

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
        self.environment = environment

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
        )
        self._setup_current(
            hosts=hosts, n_slots_per_host=self.n_slots, is_gpu=self.is_gpu, flow=flow
        )
        return hosts


def setup_mpi_env(
    run, ubf_context, all_nodes_started_timeout, interval, n_slots, is_k8s
):
    # TODO: Generalize setup to work on AWS Batch
    # NOTE: Where jobset for @kuberentes + @parallel can automate the sshd port opening,
    # AWS Batch will require security groups applied to the compute env for this.

    from metaflow import current, S3

    s3 = S3(run=run)

    # gather variables - TODO: use current
    if is_k8s:
        world_size = int(os.environ["WORLD_SIZE"])
        if ubf_context == UBF_CONTROL:
            node_index = 0
        else:
            node_index = int(os.environ["RANK"]) + 1
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
        s3.put(key_push_path, f.read())

    if ubf_context == UBF_CONTROL:
        # loop until all workers have pushed their public keys
        worker_keys_path = "%s/worker" % MPI_PUBLIC_KEY_PATH
        while True:
            try:
                paths = s3.list_paths([worker_keys_path])
                if len(paths) == world_size - 1:  # all nodes minus control
                    break
                time.sleep(interval)
            except metaflow.plugins.datatools.s3.s3.MetaflowS3NotFound:
                time.sleep(interval)
                continue

        # append all public keys to authorized_keys file
        with open(os.path.join(ssh_dir, "authorized_keys"), "a") as g:
            for p in paths:
                tail = p.url.split(MPI_PUBLIC_KEY_PATH)[-1][1:]
                obj = s3.get(os.path.join(MPI_PUBLIC_KEY_PATH, tail))
                g.write(obj.text)
            # add self to keys too
            with open(os.path.join(ssh_dir, "id_rsa.pub"), "r") as f:
                g.write(f.read())

    else:
        control_key_path = "%s/control" % MPI_PUBLIC_KEY_PATH
        while True:
            try:
                obj = s3.get(control_key_path)
                with open(os.path.join(ssh_dir, "authorized_keys"), "a") as g:
                    g.write(obj.text)
                break
            except metaflow.plugins.datatools.s3.s3.MetaflowS3NotFound:
                time.sleep(interval)
                continue
    os.chmod(os.path.join(ssh_dir, "authorized_keys"), 0o600)

    # enable passwordless ssh
    ssh_config_options = [
        "PubKeyAuthentication yes",
        "RSAAuthentication yes",  # TODO: can this be removed?
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

    # Share IPs to write the hosts file
    s3.put("%s/%s" % (HOSTFILE_IP_KEY, node_index), my_ip)
    while True:
        s3_hostfile_entry_paths = s3.list_paths([HOSTFILE_IP_KEY])
        if len(s3_hostfile_entry_paths) == world_size:
            hosts = [
                s3.get(os.path.join(*s3obj.url.split("/")[-2:])).blob.decode("utf-8")
                for s3obj in s3_hostfile_entry_paths
            ]
            break
        time.sleep(5)

    with open(HOSTFILE, "a") as f:
        for h in hosts:
            if h == my_ip:
                h = "127.0.0.1"
            f.write("%s slots=%s\n" % (h, n_slots))

    s3.close()
    return hosts


class DeepspeedException(MetaflowException):
    headline = ""

    def __init__(self, cmd):
        msg = "The Deepspeed command \n\n{}\n\nfailed to complete.".format(
            " ".join(cmd)
        )
        super(DeepspeedException, self).__init__(msg)
