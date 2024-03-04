from metaflow.unbounded_foreach import UBF_CONTROL
from metaflow.plugins.parallel_decorator import (
    ParallelDecorator,
    _local_multinode_control_task_step_func,
)
from metaflow.exception import MetaflowException
import metaflow

# from metaflow import Run as MetaflowRun
from functools import partial
from typing import List, Dict, Union, Tuple
import subprocess
import socket
import json
import time
import tempfile
import sys
import os
from io import BytesIO
from collections import namedtuple
from .exceptions import DeepspeedDatastoreNotFoundError, DeepspeedException

from metaflow import current

from .datastore import DeepspeedDatastore
from .executor import DeepspeedExecutor
from .constants import (
    HOSTFILE_IP_KEY,
    HOSTFILE,
    MPI_PUBLIC_KEY_PATH,
)

"""
Control Flow : 
1. Task pre-step sets the UBF Context. 
2. The @parallel decorator mutates the step function based because it task decorates. 
    - This mutation is essential because, we are running the actual user code differently for each task based on the type of task (worker / control).
    - The `@parallel` decorator exposes a `setup_distributed_env` method that is called by the `@deepspeed` decorator to set up the distributed environment.
3. Once `setup_distributed_env` is called, we setup the SSH connections between the nodes for MPI. 
4. 
"""


class DeepspeedDecorator(ParallelDecorator):
    name = "deepspeed"
    defaults = {"all_nodes_started_timeout": 90, "worker_polling_freq": 5}
    IS_PARALLEL = True

    requires_passwordless_ssh = (
        True  # modifies how @kubernetes runs command in docker container
    )

    def _setup_current(self, hosts, n_slots_per_host, is_gpu, flow):
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

    def task_pre_step(
        self,
        step_name,
        task_datastore,
        metadata,
        run_id,
        task_id,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
        inputs,
    ):
        self._ubf_context = ubf_context

    def setup_distributed_env(
        self,
        flow,
    ):
        "Return a list of strings of hostnames of nodes to use for MPI"
        hosts = setup_mpi_env(
            self._ubf_context,
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


def setup_ssh_keys():
    with tempfile.TemporaryDirectory() as f:
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


def setup_mpi_env(
    ubf_context,
    all_nodes_started_timeout,
    interval,
    n_slots,
    is_k8s,
    flow_datastore,
):
    # NOTE: Where jobset for @kuberentes + @parallel can automate the sshd port opening,
    # AWS Batch will require security groups applied to the compute env for this.

    datastore = DeepspeedDatastore(
        flow_datastore=flow_datastore, pathspec=current.pathspec
    )
    node_index, world_size = current.parallel.node_index, current.parallel.num_nodes

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

    """
    create ssh keys on each host: 
    1. For control task wait for the workers to write the keys in the datastore .
        - Once all keys are written, set them on the authorized hosts file.
    2. For the Worker tasks, wait for the control task key to be available. 
        - Once it is the write the public key into authorized hosts file.
    """
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
                for i in range(1, world_size):  # worker node indices start at 1
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

    # At this point nearly all workers would be running the same line of code since
    # they are all aware of the other worker's public keys.
    os.chmod(os.path.join(ssh_dir, "authorized_keys"), 0o600)

    # Instead of doing sshd restart, instead update the ssh_config file.
    # The ssh config can hold `PubKeyAuthentication` and `RSAAuthentication` set for
    # every IP address created from all the workers.

    # Core question : is the sshd service restart done because there is a change in the id_rsa/authorizedhosts
    # along with ssh_d or because there is only change in sshd_config file. If it is just the sshd change then we can get away without restarting the service
    # by just changing the ssh_config file with the right hosts.

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
    # TODO : we can share IPs along with ssh keys in one go.
    # There is no need to wait for two times to share something needed by everyone.
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
                hosts.append(datastore.get(datastore_path).blob.decode("utf-8"))
            break
        time.sleep(5)

    with open(HOSTFILE, "a") as f:
        for h in hosts:
            if h == my_ip:
                h = "127.0.0.1"
            # slots is MPI lingo which is used with deepspeed when we do passwordless ssh connections between jobs
            # slots correlates to the number of processes of user code we wish to run on that instance.
            f.write("%s slots=%s\n" % (h, n_slots))

    return hosts
