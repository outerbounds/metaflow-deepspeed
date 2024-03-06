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
from .exceptions import DeepspeedException, DatastoreKeyNotFoundError
from .datastore import DeepspeedDatastore

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

    def _construct_environment(self):
        pass

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

    def _control_node_task(self, datastore, deepspeed_args, entrypoint, entrypoint_args):
        self._exec_cmd(deepspeed_args, entrypoint, entrypoint_args)
        datastore.put(
            CONTROL_TASK_DONE_PATH, json.dumps({DEEPSPEED_JOB_COMPLETE_VAR: True})
        )
    
    def _worker_node_task(self, datastore):
        control_done = False
        while not control_done:
            time.sleep(self.worker_polling_freq)
            try:
                control_done = json.loads(
                    datastore.get(CONTROL_TASK_DONE_PATH).blob
                )[DEEPSPEED_JOB_COMPLETE_VAR]
            except DatastoreKeyNotFoundError:
                control_done = False
                continue

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
            self._worker_node_task(datastore)
            
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
