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
from .exceptions import DeepspeedDatastoreNotFoundError
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

    def get_datastore_file_location(self, key):
        return os.path.join(
            self.get_storage_root, self._flow_name, self._run_id, self._step_name, key
        )

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
            results.append(
                self.get_datastore_key_location(key)[len(keyless_root) :].strip("/")
            )
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
