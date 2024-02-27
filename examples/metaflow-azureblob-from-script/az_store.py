import os
from io import BytesIO
from itertools import starmap
from collections import namedtuple
from typing import List, Optional, Union, Tuple

from metaflow import FlowSpec
from metaflow.metaflow_config import DATATOOLS_AZUREROOT, DATASTORE_SYSROOT_AZURE
from metaflow.plugins.azure.blob_service_client_factory import get_azure_blob_service_client
from metaflow.plugins.datastores.azure_storage import AzureStorage, _AzureRootClient
from metaflow.exception import MetaflowException


AzureObject = namedtuple("AzureObject", "blob url key text")
AzureListPathResult = namedtuple("AzureListPathResult", "url key")

class AzureDatastoreNotFoundError(MetaflowException):
    headline = "Azure Datastore Object Not Found"
    def __init__(self, url):
        super(AzureDatastoreNotFoundError, self).__init__(
            "The Azure Datastore object at '%s' was not found." % url
        )


class AzureBlob(object):

    """
    An experimental class to interact with Azure Blob storage.
    It aims to mimic a slice of the Metaflow S3 client API, including these methods:
        - put: place a string or bytes object into a key in the datastore.
        - get: retrieve bytes or a string from a key in the datastore.
        - list_paths: list all objects under the datastore's `keys`.
        - put_files: modified to put raw bytes instead of files.
        - get_files (new): use to download a list of files from the datastore that were put there using put_files.
        - get_recursive: get all objects under the datastore's `keys`.
    """

    def __init__(
        self, 
        run: Optional[Union[FlowSpec, "metaflow.Run"]] = None,
        blob_path: Optional[str] = None,
        run_pathspec: Optional[str] = None,
        step_name: Optional[str] = None
    ):

        if blob_path is None and run_pathspec is None:
            raise ValueError("Either blob_path or run_pathspec must be specified in the AzureBlob instantiation.")
        if blob_path and run_pathspec:
            raise ValueError("Cannot specify both blob_path and run_pathspec in the AzureBlob instantiation.")

        # TODO: tmp
        if blob_path: 
            raise NotImplementedError("Custom blob_path is not yet implemented in AzureBlob. Pass flow_name and run_id instead.")

        self._datastore = AzureStorage()
        self._datastore_root = DATASTORE_SYSROOT_AZURE
        self._datastore.datastore_root = self._datastore_root
        self._run_pathspec = run_pathspec
        self._blob_path = blob_path
        self._step_name = step_name

    def get_datastore_key_location(self, key: Optional[str] = None):
        if self._step_name is None:
            return os.path.join(
                self._run_pathspec, key if key else ""
            ).rstrip("/")
        else:
            return os.path.join(
                self._run_pathspec, self._step_name, key if key else ""
            ).rstrip("/")

    def put(self, key: str, obj: Union[str, bytes], overwrite: bool = False):
        "Put a single object into the datastore's `key` index."

        if isinstance(obj, bytes):
            self._datastore.save_bytes(
                [(self.get_datastore_key_location(key), BytesIO(obj))],
                overwrite=overwrite,
            )
        else:
            self._datastore.save_bytes(
                [(self.get_datastore_key_location(key), BytesIO(obj.encode("utf-8")))],
                overwrite=overwrite,
            )

    def get(self, key):
        "Get a single object residing in the datastore's `key` index."
        datastore_url = self.get_datastore_key_location(key)
        keyless_root = self.get_datastore_key_location()
        with self._datastore.load_bytes([datastore_url]) as get_results:
            for key, path, meta in get_results:
                if path is not None:
                    with open(path, "rb") as f:
                        blob_bytes = f.read()
                        try:
                            return AzureObject(
                                blob=blob_bytes,
                                url=datastore_url,
                                text=blob_bytes.decode("utf-8"),
                                key=datastore_url[len(keyless_root):].strip("/")
                            )
                        except UnicodeDecodeError:
                            return AzureObject(
                                blob=blob_bytes,
                                url=datastore_url,
                                text=None,
                                key=datastore_url[len(keyless_root):].strip("/")
                            )
                else:
                    raise AzureDatastoreNotFoundError(datastore_url)

    def list_paths(self, keys: Optional[List[str]] = None):
        "List all objects in the datastore's `keys` index."
        keyless_root = self.get_datastore_key_location()
        if keys is None:
            keys = [keyless_root]
        else:
            keys = [self.get_datastore_key_location(key) for key in keys]
        results = []
        for list_content_result in self._datastore.list_content(keys):
            key_suffix = list_content_result.path[len(keyless_root):].strip("/")
            if list_content_result.is_file:
                results.append(
                    AzureListPathResult(url=list_content_result.path, key=key_suffix)
                )
            else:
                results += self.list_paths([key_suffix])
        return results

    def put_files(self, key_paths: List[Tuple[str, str]], overwrite=False):
        results = []
        for key, path in key_paths:
            with open(path, "rb") as f:
                self.put(key, f.read(), overwrite=overwrite)
            results.append(
                self.get_datastore_key_location(key)
            )
        return results

    def get_files(self, key_paths: List[Tuple[str, str]], as_binary=False):
        for key, path in key_paths:

            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
                
            if as_binary:
                with open(path, "wb") as f:
                    f.write(self.get(key).blob)
            else:
                with open(path, "w") as f:
                    f.write(self.get(key).text)

    def get_recursive(self, keys: List[str]):
        for result in self.list_paths(keys):
            if result.key:
                yield self.get(result.key)