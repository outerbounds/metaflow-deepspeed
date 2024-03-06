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
import sys
import tempfile
import os
from io import BytesIO
from collections import namedtuple
from .exceptions import DeepspeedException
from .datastore import DeepspeedDatastore
from .status_notifier import (
    TaskStatusNotifier,
    HeartbeatThread,
    wait_for_task_completion,
    HeartbeatTimeoutException,
    TaskFailedException,
)

DEEPSPEED_SUFFIX = "mf.deepspeed_datastore"


HOSTFILE_IP_KEY = "hostfile_ips"
HOSTFILE = "hostfile.txt"
DEEPSPEED_JOB_COMPLETE_VAR = "mpi_job_status"
PUBLIC_KEY_RECEIVED_VAR = "public_key_received"
CONTROL_TASK_DONE_PATH = "control"
MPI_PUBLIC_KEY_PATH = "public_keys"
DEEPSPEED_ENV_FILE = ".deepspeed_env"  # https://github.com/microsoft/DeepSpeed/blob/24f20ef0a105d32f6085fe0d3b1c2f9324a6262c/docs/_tutorials/getting-started.md?plain=1#L230-L254
DEEPSPEED_SUFFIX = "mf.deepspeed_datastore"


def _get_path(local_output_dir, cloud_output_dir, node_index):
    filepath_tuples = []
    for path, subdirs, files in os.walk(local_output_dir):
        for fname in files:
            filepath_tuples.append(
                (
                    f"{cloud_output_dir}/{str(node_index)}/{os.path.relpath(os.path.join(path, fname), local_output_dir)}",
                    os.path.join(path, fname),
                )
            )
    return filepath_tuples


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
        self._heartbeat_thread = None  # This is only set on the control task

    def _exec_cmd(
        self,
        deepspeed_args: Union[List[str], Dict[str, str]] = [],
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = [],
    ):
        """
        This will ONLY be executed by the control node.
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
                if (
                    k == "METAFLOW_INIT_SCRIPT"
                ):  # TODO: Remove the MF_PARALLEL variables from here since it will be passed down to workers via SSH.
                    continue  # Don't pass these to deepspeed. Because of how deepspeed reads env vars and sets them in runner commands.
                else:
                    json_string = json.dumps(v)
                    f.write(f"{k}={json_string}\n")

        # Launch the Deepspeed run.
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,  # pipe to Metaflow stdout. TODO: How to handle progress bar buffering like TQDM.
            stderr=subprocess.PIPE,
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
                return False, process.stderr.read().decode("utf-8")
            return True, None

    def _resolve_storage_paths(
        self, push_results_dir_to_cloud, datastore, local_output_dir, cloud_output_dir
    ) -> str:
        if push_results_dir_to_cloud and (
            datastore._backend.TYPE != "s3" and datastore._backend.TYPE != "azure"
        ):
            raise MetaflowException(
                "current.deepspeed.run must use S3 or Azure Blob Storage as a datastore if push_results_dir_to_cloud is True. You are using %s."
                % datastore._backend.TYPE
            )
        elif push_results_dir_to_cloud:
            # TODO: Annoying place for this check. Consider moving the S3 push args into the decorator itself, so can be checked at flow init instead.
            if local_output_dir is None:
                raise MetaflowException(
                    "current.deepspeed.run must specify local_output_dir if push_results_dir_to_s3 is True"
                )
            elif cloud_output_dir is None:
                if local_output_dir.startswith("/"):
                    cloud_output_dir = local_output_dir[1:]
                else:
                    cloud_output_dir = local_output_dir
        return cloud_output_dir

    def _control_node_task(
        self, datastore: DeepspeedDatastore, deepspeed_args, entrypoint, entrypoint_args
    ):
        """
        The control task will run the `deepspeed` command using all the arguments provided
        to the `DeepSpeedExecutor.run` method. The control task will also publish heartbeats
        to the datastore every 3 seconds to indicate that it is still running. These heartbeats
        are used by the worker tasks to monitor the control task's status and essentially keep the workers alive.

        Since the `DeepSpeedExecutor` is an abstraction over the `deepspeed` command, we will capture the stderr
        when we call the `deepspeed` command and pass it down to the `DeepspeedException` if the `deepspeed` command fails.
        """
        # set the status that the control node is Up
        _status_notifier = TaskStatusNotifier(datastore)
        _status_notifier.running(0)
        # start the heartbeat thread that writes the status to the datastore every 3 seconds
        self._heartbeat_thread = HeartbeatThread(
            _status_notifier, node_index=0, heartbeat_interval=3
        )
        self._heartbeat_thread.start()
        # run the deepspeed command
        success_status, stderr = self._exec_cmd(
            deepspeed_args, entrypoint, entrypoint_args
        )
        # stop the heartbeat thread upon failure / success of control task.
        self._heartbeat_thread.stop()
        # sets the status of the control node to finished or failed
        if success_status:
            _status_notifier.finished(0)
        else:
            _status_notifier.failed(0)
            msg = f"The `deepspeed` command running on the control task has crashed. \n\n[stderr]: {str(stderr)}"
            raise DeepspeedException(msg)

    def _worker_node_task(
        self, datastore: DeepspeedDatastore, node_index: int, heartbeat_timeout=60 * 10
    ):
        """
        The worker tasks will poll for the control task's heartbeat and do nothing else.
        Given that the tasks are run using MPI/ssh, the control task ends up controlling the worker directly (via ssh)
        and metaflow doesn't need to do anything else.

        Any failure in the worker's entry-point script will result in the failure at the control task level.
        The only way a worker will end up failing will be :
            - The control task fails (Which can happen the worker's entry-point script fails, resulting in the control task failing.)
            - if the duration of the last heartbeat crosses the `heartbeat_timeout`

        Hence the control flow on the worker only monitors the heartbeat/statues set from the control task.
        """
        # TODO : Make heartbeat timeout configurable
        _status_notifier = TaskStatusNotifier(datastore)
        # Worker task statuses are only for bookkeeping.
        # They are not used by the control task in any way.
        _status_notifier.running(node_index)
        try:
            # Poll the control task's heartbeat and fail if control task fails
            # or if the heartbeat interval crosses the threshold.
            wait_for_task_completion(
                _status_notifier, node_index=0, heartbeat_timeout=heartbeat_timeout
            )
            _status_notifier.finished(node_index)
        except HeartbeatTimeoutException:
            _status_notifier.failed(node_index)
            raise DeepspeedException(
                f"Control task heartbeat timed out. Control task has not published a heartbeat for {heartbeat_timeout} seconds."
            )
        except TaskFailedException:
            _status_notifier.failed(node_index)
            raise DeepspeedException("Control task reported failure.")

    def run(
        self,
        deepspeed_args: Union[List[str], Dict[str, str]] = [],
        entrypoint: str = None,
        entrypoint_args: Union[List[str], Dict[str, str]] = [],
        push_results_dir_to_cloud: bool = False,
        local_output_dir: str = None,
        cloud_output_dir: str = None,
    ) -> None:
        from metaflow import current

        node_index = current.parallel.node_index  # assumes parallel
        datastore = DeepspeedDatastore(
            flow_datastore=self._flow_datastore, pathspec=current.pathspec
        )

        # Resolve storage paths
        cloud_output_dir = self._resolve_storage_paths(
            push_results_dir_to_cloud, datastore, local_output_dir, cloud_output_dir
        )

        # Run the distributed job
        if node_index == 0:  # control node
            self._control_node_task(
                datastore, deepspeed_args, entrypoint, entrypoint_args
            )
        else:  # worker node
            self._worker_node_task(datastore, node_index)

        # Push results to S3
        if push_results_dir_to_cloud:
            if not os.path.exists(local_output_dir):
                print(
                    f"Deepspeed process completed, and local_output_dir `{local_output_dir}` does not exist, skipping push to S3.",
                    file=sys.stderr,
                )
                return
            paths = datastore.put_files(
                _get_path(local_output_dir, cloud_output_dir, node_index)
            )
            print(
                f"Pushed {len(paths)} files to {datastore._backend.TYPE} at {cloud_output_dir}",
                file=sys.stderr,
            )
            return datastore.get_datastore_file_location(cloud_output_dir)

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
