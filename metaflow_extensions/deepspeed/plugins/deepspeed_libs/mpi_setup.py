from metaflow.unbounded_foreach import UBF_CONTROL
import subprocess
import socket
import time
import os
from .exceptions import (
    SSHKeyGenException,
    SSHServiceRestartException,
    SSHScanUnsuccessfulException,
)
from metaflow import current

from .datastore import DeepspeedDatastore, task_sync_barrier
from .constants import (
    HOSTFILE_IP_KEY,
    HOSTFILE,
    MPI_PUBLIC_KEY_PATH,
)


class KeyPaths:
    Control = lambda: f"{MPI_PUBLIC_KEY_PATH}/control"
    Worker = lambda x: f"{MPI_PUBLIC_KEY_PATH}/worker/{x}"
    Host = lambda x: f"{HOSTFILE_IP_KEY}/{x}"


def create_and_push_ssh_keys(datastore: DeepspeedDatastore, ubf_context, node_index):
    """
    Create SSH keys and push the public key to the datastore.
    Also write the IP of the host to the datastore.
    """
    if ubf_context == UBF_CONTROL:
        key_push_path = KeyPaths.Control()
    else:
        key_push_path = KeyPaths.Worker(node_index)

    ssh_dir = os.path.expanduser("~/.ssh")
    if not os.path.exists(ssh_dir):
        os.makedirs(ssh_dir)

    os.chmod(ssh_dir, 0o700)  # TODO : Fix me : rest permissioning correct.
    # TODO : handle case where id_rsa is present
    # TODO: handle case where

    _key_name = "id_rsa"
    # Generate host key if it doesn't exist! NEVER overwrite existing ssh keys of some machine
    if not os.path.exists(os.path.join(ssh_dir, _key_name)):
        try:
            result = subprocess.run(
                [
                    "ssh-keygen",
                    "-t",
                    "rsa",
                    "-f",
                    os.path.join(ssh_dir, _key_name),
                    "-N",
                    "",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise SSHKeyGenException(
                cmd=e.cmd,
                exception=e.stderr,
                worker_type=ubf_context,
                node_index=node_index,
            )

    if not os.path.exists(os.path.join(ssh_dir, "id_rsa.pub")):
        raise SSHKeyGenException(
            cmd="ssh-keygen",
            exception="Public key not found",
            worker_type=ubf_context,
            node_index=node_index,
        )

    # Move public key to S3
    with open(os.path.join(ssh_dir, "id_rsa.pub"), "r") as f:
        datastore.put(key_push_path, f.read())

    # Write the IP of the host to the datastore
    host_key = KeyPaths.Host(node_index)
    datastore.put(host_key, get_my_ip())

    return ssh_dir


def _control_task_ssh_setup(
    datastore,
    ssh_dir,
    world_size,
    max_wait_time=600,
    frequency=0.1,
):
    # All worker indexes start from 1
    _worker_keys = [KeyPaths.Worker(i) for i in range(1, world_size)]
    _lock_args = {
        "description": "Waiting for worker tasks to write public keys to datastore",
        "max_wait_time": max_wait_time,
        "frequency": frequency,
    }
    with task_sync_barrier(
        "ControlTaskPublicKeySync", datastore, _worker_keys, **_lock_args
    ) as data:
        # append all public keys to authorized_keys file
        with open(os.path.join(ssh_dir, "authorized_keys"), "a") as g:
            for p in _worker_keys:
                obj = data[p]
                g.write(obj.text)
            # add self to keys too
            with open(os.path.join(ssh_dir, "id_rsa.pub"), "r") as f:
                g.write(f.read())


def _worker_task_ssh_setup(datastore, ssh_dir, max_wait_time=600, frequency=0.1):
    control_key_path = KeyPaths.Control()
    _lock_args = {
        "description": "Waiting for control task to write public key to datastore",
        "max_wait_time": max_wait_time,
        "frequency": frequency,
    }
    with task_sync_barrier(
        "WorkerTaskPublicKeySync", datastore, [control_key_path], **_lock_args
    ) as data:
        with open(os.path.join(ssh_dir, "authorized_keys"), "a") as g:
            g.write(data[control_key_path].text)


def update_ssh_config_and_restart_sshd(ssh_config_options, ubf_context, node_index, use_sudo=False):
    # This is un-used but can be used later if necessary.
    with open("/etc/ssh/sshd_config", "a") as f:
        f.write("\n".join(ssh_config_options))

    # TODO : Here we require sudo access and sshd service restart. This should be done better!
    try:
        result = subprocess.run(
            (["sudo"] if use_sudo else []) + ["service", "ssh", "restart"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise SSHServiceRestartException(
            cmd=e.cmd,
            exception=e.stderr,
            worker_type=ubf_context,
            node_index=node_index,
        )

    return True


def get_my_ip():
    return socket.gethostbyname(socket.gethostname())


def host_file_sync(datastore, world_size, max_wait_time=600, frequency=0.1):
    host_key_paths = [KeyPaths.Host(i) for i in range(world_size)]
    description = "Waiting for all workers to write their IPs to the datastore"
    with task_sync_barrier(
        "HostFileSync",
        datastore,
        host_key_paths,
        max_wait_time=max_wait_time,
        frequency=frequency,
        description=description,
    ) as data:
        hosts = []
        for p in host_key_paths:
            hosts.append(data[p].text)
        return hosts


def _write_hostfile(hosts, n_slots):
    my_ip = get_my_ip()
    with open(HOSTFILE, "a") as f:
        for h in hosts:
            if h == my_ip:
                h = "127.0.0.1"
            # slots is MPI lingo which is used with deepspeed when we do passwordless ssh connections between jobs
            # slots correlates to the number of processes of user code we wish to run on that instance.
            f.write("%s slots=%s\n" % (h, n_slots))


def _scan_cmd(host):
    return ["ssh-keyscan", "-H", host]


def scan_one_host(host, max_retries=5, retry_delay=5):
    curr_retry = 0
    error_log = None
    while curr_retry < max_retries:
        try:
            result = subprocess.run(
                _scan_cmd(host),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            # TODO [LOGGING] : make the error visible to the user.
            curr_retry += 1
            if e.stderr:
                error_log = e.stderr.decode("utf-8")
            time.sleep(retry_delay)
            continue
        else:
            return result.stdout.decode("utf-8"), None
    return None, error_log  # Only when it failed all tries.


def scan_all_hosts(hosts, ubf_context, node_index, max_retries=5, retry_delay=5):
    host_keys = []
    for host in hosts:
        output, error_log = scan_one_host(host, max_retries, retry_delay)
        if output is None:
            raise SSHScanUnsuccessfulException(error_log, ubf_context, node_index, host)
        host_keys.append(output)
    return host_keys


def setup_known_hosts(hosts, ubf_context, node_index):
    """
    create the known_hosts file with the public keys of all the hosts.
    """
    my_ip = get_my_ip()
    _hosts = [h for h in hosts if h != my_ip]
    _hosts += ["127.0.0.1"]
    host_keys = scan_all_hosts(
        _hosts, ubf_context, node_index, max_retries=5, retry_delay=5
    )
    known_hosts_path = os.path.expanduser("~/.ssh/known_hosts")
    with open(known_hosts_path, "a") as f:
        for key in host_keys:
            f.write(key)


def setup_mpi_env(
    flow_datastore,
    ubf_context: str = UBF_CONTROL,
    all_nodes_started_timeout: int = 600,
    n_slots: int = 1,
    polling_frequency: float = 0.1,
    use_sudo: bool = False,
):
    """
    1. create ssh keys on each host
    2. based on the type of task do the following :
        - For control task wait for the workers to write the keys in the datastore .
            - Once all keys are written, set them on the authorized hosts file.
        - For the Worker tasks, wait for the control task key to be available.
            - Once it is the write the public key into authorized hosts file.
    """
    # NOTE: Where jobset for @kuberentes + @parallel can automate the sshd port opening,
    # AWS Batch will require security groups applied to the compute env for this.

    datastore = DeepspeedDatastore(
        flow_datastore=flow_datastore, pathspec=current.pathspec
    )
    node_index, world_size = current.parallel.node_index, current.parallel.num_nodes

    ssh_dir = create_and_push_ssh_keys(datastore, ubf_context, node_index)

    if ubf_context == UBF_CONTROL:
        _control_task_ssh_setup(
            datastore,
            ssh_dir,
            world_size,
            max_wait_time=all_nodes_started_timeout,
            frequency=polling_frequency,
        )
    else:
        _worker_task_ssh_setup(
            datastore,
            ssh_dir,
            max_wait_time=all_nodes_started_timeout,
            frequency=polling_frequency,
        )

    # At this point nearly all workers would be running the same line of code since
    # they are all aware of the other worker's public keys.
    # Todo : See if we can find a way to get rid of permissioning setting here.
    os.chmod(os.path.join(ssh_dir, "authorized_keys"), 0o600)

    # Instead of doing sshd restart, instead update the ssh_config file.
    # The ssh config can hold `PubKeyAuthentication` and `RSAAuthentication` set for
    # every IP address created from all the workers.

    # Core question : is the sshd service restart done because there is a change in the id_rsa/authorizedhosts
    # along with ssh_d or because there is only change in sshd_config file. If it is just the sshd change then we can get away without restarting the service
    # by just changing the ssh_config file with the right hosts.

    # enable passwordless ssh and done with SUDO!
    # This is not the best way but we can contribute more time
    # if the usage of this is more frequent and requires rootless environments.
    update_ssh_config_and_restart_sshd(
        [
            "PubKeyAuthentication yes",
            "RSAAuthentication yes",
        ],
        ubf_context,
        node_index,
        use_sudo=use_sudo
    )
    # Write all the IP's shared by the workers to the hostfile.

    hosts = host_file_sync(
        datastore,
        world_size,
        max_wait_time=all_nodes_started_timeout,
        frequency=polling_frequency,
    )
    _write_hostfile(hosts, n_slots)

    # At this point all hosts should be accessible so accordingly
    # setup the known hosts file.
    setup_known_hosts(hosts, ubf_context, node_index)

    return hosts
