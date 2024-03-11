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
from metaflow import current

from .executor import DeepspeedExecutor
from .mpi_setup import setup_mpi_env

"""
Control Flow : 
1. Task pre-step sets the UBF Context. 
2. The @parallel decorator mutates the step function based because it task decorates. 
    - This mutation is essential because, we are running the actual user code differently for each task based on the type of task (worker / control).
    - The `@parallel` decorator exposes a `setup_distributed_env` method that is called by the `@deepspeed` decorator to set up the distributed environment.
3. Once `setup_distributed_env` is called, we sets up the SSH connections between the nodes for MPI. 
4. Use code can now call `current.deepspeed.run` to run the training script with deepspeed. 
    - `current.deepspeed.run` only executes the entry-point script on the control task
    - The worker tasks are controlled from the control task using MPI with SSH. 
"""


class DeepspeedDecorator(ParallelDecorator):
    name = "deepspeed"
    # TODO : Safe rename `all_nodes_started_timeout` to something more intuitive.
    defaults = {"all_nodes_started_timeout": 600, "worker_polling_freq": 0.5}
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

        for deco in decos: 
            if deco.name in ["resources", "kubernetes", "batch"]:
                if deco.attributes['gpu']:
                    self.is_gpu = True
                    self.n_slots = deco.attributes['gpu']
                else:
                    self.is_gpu = False
                    self.n_slots = deco.attributes['cpu']

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
            self.n_slots,
            self.flow_datastore,
        )
        self._setup_current(
            hosts=hosts, n_slots_per_host=self.n_slots, is_gpu=self.is_gpu, flow=flow
        )
        return hosts
