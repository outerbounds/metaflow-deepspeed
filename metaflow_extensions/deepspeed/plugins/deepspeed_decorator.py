from metaflow.unbounded_foreach import UBF_CONTROL
from metaflow.plugins.parallel_decorator import ParallelDecorator, _local_multinode_control_task_step_func
from metaflow.exception import MetaflowException
import metaflow
from functools import partial
import subprocess
import socket
import json
import time
import sys
import os
import io

HOSTFILE_IP_KEY = 'hostfile_ips'
HOSTFILE = 'hostfile.txt'
DEEPSPEED_JOB_COMPLETE_VAR = "mpi_job_status"
PUBLIC_KEY_RECEIVED_VAR = "public_key_received"
CONTROL_TASK_DONE_PATH = 'control'
MPI_PUBLIC_KEY_PATH = 'public_keys'

class DeepspeedExecutor:

    def __init__(self, hosts, n_slots_per_host=1, is_gpu=False) -> None:
        self.is_gpu = is_gpu
        self.n_slots_per_host = n_slots_per_host
        self.hosts = [h for h in hosts if h != socket.gethostbyname(socket.gethostname())] + ["127.0.0.1"]
        self._scan_all_hosts()

    def _exec_cmd(self, deepspeed_args=[], entrypoint=None, entrypoint_args=[], deepspeed_config=None):
        """
        `deepspeed_args` : List[str] - arguments to pass to `exe`
        `entrypoint`: str - program to run, such as compiled binary or python script
        `entrypoint_args`: List[str] - arguments to pass after `entrypoint`
        `deepspeed_config`: ... TODO json thing
        """

        cmd = ["deepspeed"]

        if "--hostfile" not in deepspeed_args:
            deepspeed_args.extend(["--hostfile", HOSTFILE])
        if "--num_nodes" not in deepspeed_args:
            deepspeed_args.extend(["--num_nodes", str(len(self.hosts))])
        if self.is_gpu and "--num_gpus" not in deepspeed_args:
            deepspeed_args.extend(["--num_gpus", str(self.n_slots_per_host)])
        if "--master_addr" in deepspeed_args:
            raise MetaflowException("Do not specify the --master_addr in your current.run.deepspeed args. Metaflow will set this for you.")
            
        my_ip = socket.gethostbyname(socket.gethostname())
        deepspeed_args.extend(["--master_addr", my_ip])

        cmd.extend(deepspeed_args)

        if entrypoint is not None:
            cmd.append(entrypoint)
        else:
            raise MetaflowException("current.deepspeed.run(..., entrypoint=<SCRIPT>, ...) arg must be specified.")

        cmd.extend(entrypoint_args)
        
        print(" ".join(cmd))
        # TODO: Remove this print. Should log with MF? 
        # It is nice for debugging on workstation to have cmd in logs. 
        # Could attach to flow.

        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
                while process.poll() is None:
                    stdout = process.stdout.read1()
                    try:
                        text = stdout.decode("utf-8")
                    except UnicodeDecodeError:
                        # TODO: This print feels bad, maybe remove - even better, 
                        # figure out how to handle the edge decoding cases gracefully.
                        # print("UnicodeDecodeError, skipping decoding of problematic bytes: %s" % stdout)
                        text=''
                    
                    print(text, end='', flush=True)
                    # TODO (Eddie): what is strat for dynamic cards? stuff `text` somewhere?

        except subprocess.CalledProcessError as e:
            print(e.stdout)
            raise e
    
    def run(self, deepspeed_args=[], entrypoint=None, entrypoint_args=[], deepspeed_config=None):
        self._exec_cmd(deepspeed_args, entrypoint, entrypoint_args, deepspeed_config)

    def _scan_cmd(self, host):
        return ["ssh-keyscan", "-H", host]

    def _scan_all_hosts(self):
        for host in self.hosts:
            try: 
                result = subprocess.run(self._scan_cmd(host), stdout = subprocess.PIPE, stderr = subprocess.STDOUT, check=True)
                if result.returncode != 0:
                    raise Exception("Error adding host to known_hosts: " + result.stdout.decode('utf-8'))
                else:
                    with open(os.path.expanduser("~/.ssh/known_hosts"), 'a') as f:
                        f.write(result.stdout.decode('utf-8'))
            except subprocess.CalledProcessError as e:
                print(e.stdout)
                raise e


class DeepspeedDecorator(ParallelDecorator):

    name = "deepspeed"
    defaults = {"all_nodes_started_timeout": 90, "worker_polling_freq": 5}
    IS_PARALLEL = True

    # Is this an anti-pattern?
    # Could not find a better way without modifying the base @kubernetes + @parallel implementation
    # based on MPI-like decorators that need passwordless ssh and to be running sshd as daemon.
    requires_passwordless_ssh = True

    def _setup_current(self, hosts, n_slots_per_host, is_gpu):
        from metaflow import current
        current._update_env({"deepspeed" : DeepspeedExecutor(hosts, n_slots_per_host, is_gpu)})

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
            if deco.name in ['resources', 'kubernetes', 'batch']:
                compute_deco_attrs = compute_resource_attributes(decos, deco, {"cpu": "1", "gpu": "0"})
                try:
                    self.n_slots = int(compute_deco_attrs['gpu'])
                    self.is_gpu = True
                except KeyError:
                    self.n_slots = int(compute_deco_attrs['cpu'])
                    self.is_gpu = False
                if not self.n_slots > 0:
                    self.n_slots = int(compute_deco_attrs['cpu'])
                    self.is_gpu = False
                break

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):

        # TODO: Investigate better patterns for submitting step_func code on control node and looping the workers.
        # The pattern in this function works for running @ray_parallel, @mpi, @deepspeed, etc.
        # Seems there should be a better, more elegant way to not always run the same exact code in the @parallel context.
        # https://github.com/Netflix/metaflow/blob/d01ae37c03363be0289fbe02d10d26cf4fb3bc1b/metaflow/task.py#L571

        from metaflow import current, S3

        if not os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local":
            s3 = S3(run=flow)

        def _worker_heartbeat(polling_freq=self.attributes["worker_polling_freq"], var=DEEPSPEED_JOB_COMPLETE_VAR):
            control_done = False
            while not control_done:
                time.sleep(polling_freq)
                try:
                    control_done = json.loads(s3.get(CONTROL_TASK_DONE_PATH).blob)[var]
                except metaflow.plugins.datatools.s3.s3.MetaflowS3NotFound:
                    control_done = False
                    continue
                
        def _control_wrapper(step_func, flow, var=DEEPSPEED_JOB_COMPLETE_VAR):
            step_func()
            s3.put(CONTROL_TASK_DONE_PATH, json.dumps({var: True}))

        def _empty_worker_task():
            pass

        if os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local":
            if ubf_context == UBF_CONTROL:
                env_to_use = getattr(self.environment, "base_env", self.environment)
                self._setup_current(hosts=["127.0.0.1"], n_slots_per_host=1, is_gpu=False)
                return partial(
                    _local_multinode_control_task_step_func,
                    flow,
                    env_to_use, 
                    step_func,
                    retry_count,
                )
            return partial(_empty_worker_task)
        else:
            hosts = self.setup_distributed_env(flow, ubf_context, n_slots=self.n_slots)
            self._setup_current(hosts=hosts, n_slots_per_host=self.n_slots, is_gpu=self.is_gpu)
            if ubf_context == UBF_CONTROL:
                return partial(_control_wrapper, step_func=step_func, flow=flow)
            return partial(_worker_heartbeat)

    def setup_distributed_env(self, flow, ubf_context, n_slots = 1):
        "Return a list of strings of hostnames of nodes to use for MPI"
        return setup_mpi_env(flow, ubf_context, self.attributes["all_nodes_started_timeout"], self.attributes["worker_polling_freq"], n_slots, self.is_k8s)


def setup_mpi_env(run, ubf_context, all_nodes_started_timeout, interval, n_slots, is_k8s):

    # TODO: Generalize setup to work on AWS Batch
    # NOTE: Where jobset for @kuberentes + @parallel can automate the sshd port opening, 
    # AWS Batch will require security groups applied to the compute env for this.

    from metaflow import current, S3
    s3 = S3(run=run)

    # gather variables
    if is_k8s:
        world_size = int(os.environ["WORLD_SIZE"])
        if ubf_context == UBF_CONTROL:
            node_index = 0
        else:
            node_index = int(os.environ["RANK"]) + 1
    else:  # is_batch
        world_size = int(os.environ["AWS_BATCH_JOB_NUM_NODES"])
        node_index = int(os.environ["AWS_BATCH_JOB_NODE_INDEX"])

        # TODO: DELETE THIS
        # set vars for deepspeed 
        # https://github.com/microsoft/DeepSpeed/blob/a855405e0b85fdc8346ff5fc0ab4085f18d95a9a/deepspeed/comm/comm.py#L150
        # os.environ['WORLD_SIZE'] = str(world_size)
        # os.environ['RANK'] = str(node_index)

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
        ["ssh-keygen", "-t", "rsa", "-f", os.path.join(ssh_dir, "id_rsa"), '-N', ''], 
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        check=True
    )
    assert result.returncode == 0, "Error generating host key"
    # Move public key to S3
    with open(os.path.join(ssh_dir, "id_rsa.pub"), 'r') as f:
        s3.put(key_push_path, f.read())

    if ubf_context == UBF_CONTROL:
        
        # loop until all workers have pushed their public keys
        worker_keys_path = "%s/worker" % MPI_PUBLIC_KEY_PATH
        while True:
            try:
                paths = s3.list_paths([worker_keys_path])
                if len(paths) == world_size - 1: # all nodes minus control
                    break
                time.sleep(interval)
            except metaflow.plugins.datatools.s3.s3.MetaflowS3NotFound:
                time.sleep(interval)
                continue
        
        # append all public keys to authorized_keys file
        with open(os.path.join(ssh_dir, "authorized_keys"), 'a') as g:
            for p in paths:
                tail = p.url.split(MPI_PUBLIC_KEY_PATH)[-1][1:]
                obj = s3.get(os.path.join(MPI_PUBLIC_KEY_PATH, tail))
                g.write(obj.text)
            # add self to keys too
            with open(os.path.join(ssh_dir, "id_rsa.pub"), 'r') as f:
                g.write(f.read())

    else:
        control_key_path = "%s/control" % MPI_PUBLIC_KEY_PATH
        while True:
            try:
                obj = s3.get(control_key_path)
                with open(os.path.join(ssh_dir, "authorized_keys"), 'a') as g:
                    g.write(obj.text)
                break
            except metaflow.plugins.datatools.s3.s3.MetaflowS3NotFound:
                time.sleep(interval)
                continue
    os.chmod(os.path.join(ssh_dir, "authorized_keys"), 0o600)

    # enable passwordless ssh
    ssh_config_options = [
        "PubKeyAuthentication yes",
        "RSAAuthentication yes" # TODO: can this be removed?
    ]
    with open('/etc/ssh/sshd_config', 'a') as f:
        f.write('\n'.join(ssh_config_options))
    result = subprocess.run(["sudo", "service", "ssh", "restart"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, "Error restarting sshd"

    # Share IPs to write the hosts file
    s3.put("%s/%s" % (HOSTFILE_IP_KEY, node_index), my_ip)
    while True: 
        s3_hostfile_entry_paths = s3.list_paths([HOSTFILE_IP_KEY])
        if len(s3_hostfile_entry_paths) == world_size:
            hosts = [
                s3.get(os.path.join(*s3obj.url.split('/')[-2:])).blob.decode('utf-8')
                for s3obj in s3_hostfile_entry_paths
            ]
            break
        time.sleep(5)

    with open(HOSTFILE, 'a') as f:
        for h in hosts:
            if h == my_ip:
                h = "127.0.0.1"
            f.write("%s slots=%s\n" % (h, n_slots))

    s3.close()
    return hosts