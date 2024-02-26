from argparse import ArgumentParser
import time
from metaflow import Run
from random import choice
from metaflow.plugins.azure.blob_service_client_factory import get_azure_blob_service_client
from metaflow.metaflow_config import DATATOOLS_AZUREROOT
from urllib.parse import urlparse
from itertools import starmap
from azure.storage.blob.aio import BlobClient


class AzureBlob(object):

    """
    An experimental class to interact with Azure Blob storage.
    It is intended to mimic a small slice of the Metaflow S3 client API.
    """

    def init(
        self, 
        run: Optional[FlowSpec, "Run"] = None,
        blob_container_name: Optional[str] = None,
        flow_name: str = None,
        run_id: str = None
    ):

        self._blob_storage_act_service = get_azure_blob_service_client()

        if run:
            if blob_container_name is not None:
                raise ValueError("Cannot specify both run and blob_container_name in AzureBlob instantiaton.")
            if DATATOOLS_AZURE_ROOT is None:
                raise ValueError("DATATOOLS_AZUREROOT is not configured when trying to use Azure Blob storage.")
            if not run_id:
                raise ValueError("run_id is required when using Azure Blob storage.")
            if not flow_name:
                raise ValueError("flow_name is required when using Azure Blob storage.")
            self._blob_container_name = os.path.join(DATATOOLS_AZUREROOT, flow_name, run_id)
        elif blob_root:
            self._blob_container_name = blob_container_name
        else:
            raise ValueError("Either run or blob_container_name must be specified in the AzureBlob instantiation.")

    def get(self, key):
        blob = storage_account_service.get_blob_client(container=self._blob_container_name, blob=blob_name)

    def get_recursive(self, keys):
        pass

    def _url(self, key):
        pass

    def _read_many_files(self, method, urls):
        pass

    def list_paths(self, keys):

        def _list(keys):
            if keys is None:
                keys = [None]
            urls = ((self._url(key).rstrip("/") + "/", None) for key in keys)
            res = self._read_many_files("list", urls)
            for s3prefix, s3url, size in res:
                if size:
                    yield s3prefix, s3url, None, int(size)
                else:
                    yield s3prefix, s3url, None, None

        return list(starmap(AzureObject, _list(keys)))

    def put(self, key, value):
        pass

    def put_files(self, files):
        pass

    def close(self):
        """
        Delete all temporary files downloaded in this context.
        """
        pass
        # try:
        #     if not debug.s3client:
        #         if self._tmpdir:
        #             shutil.rmtree(self._tmpdir)
        #             self._tmpdir = None
        # except:
        #     pass


def do_metaflow_operation(run_id, flow_name, global_rank):

    # business logic.
    animal = choice(["hippopotamus", "giraffe", "monkey", "lion", "tiger", "bear"])

    # push results using Metaflow S3.
    s3 = S3(run=Run(f"{flow_name}/{run_id}"))
    res = s3.put(
        f"output_{global_rank}.txt",
        f"Process {global_rank} wants a {animal} for Christmas.",
    )
    s3.close()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument("--run_id", type=str, default=None, help="Metaflow run id")
    parser.add_argument(
        "--flow_name", type=str, default=None, help="Metaflow flow_name"
    )
    args = parser.parse_args()

    time.sleep(2)
    from deepspeed import comm as dist
    from deepspeed import init_distributed

    init_distributed(dist_backend="gloo")
    do_metaflow_operation(
        run_id=args.run_id, flow_name=args.flow_name, global_rank=dist.get_rank()
    )
    time.sleep(2)
