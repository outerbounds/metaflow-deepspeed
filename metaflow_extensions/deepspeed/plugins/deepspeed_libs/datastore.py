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
import os
from io import BytesIO
from collections import namedtuple
from contextlib import contextmanager
from .exceptions import DatastoreKeyNotFoundError, WaitLockFailed
from .constants import DEEPSPEED_SUFFIX

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

    def get_datastore_key_location(self):
        return os.path.join(
            self.get_storage_root, self._flow_name, self._run_id, self._step_name
        )

    def get_datastore_file_location(self, key):
        return os.path.join(self.get_datastore_key_location(), key)

    def put(self, key: str, obj: Union[str, bytes], overwrite: bool = False):
        "Put a single object into the datastore's `key` index."
        _save_object = None
        if isinstance(obj, bytes):
            _save_object = BytesIO(obj)
        else:
            _save_object = BytesIO(obj.encode("utf-8"))

        self._backend.save_bytes(
            [(self.get_datastore_file_location(key), _save_object)],
            overwrite=overwrite,
        )

    def put_files(self, key_paths: List[Tuple[str, str]], overwrite=False):
        keyless_root = self.get_datastore_key_location()
        results = []
        for key, path in key_paths:
            with open(path, "rb") as f:
                self.put(key, f.read(), overwrite=overwrite)
            results.append(self.get_datastore_file_location(key))
        return results

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
                    raise DatastoreKeyNotFoundError(datastore_url)

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


def _best_effort_read_key(datastore: DeepspeedDatastore, key: str):
    """
    Read a key from the datastore, but do not raise an error if the key is not found.
    """
    try:
        return datastore.get(key)
    except DatastoreKeyNotFoundError:
        return None


def _best_effort_get_keys(datastore: DeepspeedDatastore, keys: List[str]):
    """
    Get the keys from the datastore, but do not raise an error if the keys are not found.
    """
    results = {}
    not_found_keys = []
    for key in keys:
        data = _best_effort_read_key(datastore, key)
        if data:
            results[key] = data
        else:
            not_found_keys.append(key)
    return results, not_found_keys


def wait_for_key_data(
    datastore: DeepspeedDatastore,
    keys: List[str],
    max_wait_time: float = 600,
    frequency=0.1,
) -> Dict[str, DeepspeedDatastoreBlob]:
    """
    Wait for the keys to be available in the datastore.
    If the keys are not available after `max_wait_time` seconds, raise an error.
    """
    start = time.time()
    exit_condition = lambda: time.time() - start > max_wait_time
    _current_keys = keys.copy()
    main_data = {}
    while not exit_condition():
        data, _ = _best_effort_get_keys(datastore, _current_keys)
        # if all keys are found, return the data
        if len(main_data) == len(keys):
            return main_data
        main_data.update(data)
        # update the current keys to wait for the remaining keys
        _current_keys = list(set(keys) - set(main_data.keys()))
        time.sleep(frequency)
    raise DatastoreKeyNotFoundError(
        f"Keys {keys} were not found in the datastore after {max_wait_time} seconds."
    )


@contextmanager
def task_sync_barrier(
    lock_name,
    datastore: DeepspeedDatastore,
    keys: List[str],
    max_wait_time=600,
    frequency=0.1,
    description=None,
):
    """
    A context manager that waits for keys to be written to the datastore and acts like a distributed-barrier.
    When multiple tasks are running in parallel, this context manager can be used to ensure that all tasks
    can wait on a certain keys to be written to the datastore. If the keys are not written to the datastore
    after `max_wait_time` seconds, a `WaitLockFailed` error is raised.

    This way only once all the keys are written to the datastore, the tasks will proceed.

    Args:
        lock_name (str): The name of the lock. Used for debugging purposes.
        datastore (DeepspeedDatastore)
        keys (List[str]): The keys to wait for in the datastore.
        max_wait_time (float): The maximum time to wait for the keys to be written to the datastore.
        frequency (float): The frequency to check the datastore for the keys.
        description (str): A description of the lock. Used for debugging purposes.
    """
    try:
        data = wait_for_key_data(datastore, keys, max_wait_time, frequency)
    except DatastoreKeyNotFoundError:
        raise WaitLockFailed(lock_name, description)
    else:
        yield data
