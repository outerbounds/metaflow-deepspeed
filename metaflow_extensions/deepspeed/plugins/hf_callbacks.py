from typing import Dict, Union
import os
import tarfile
from metaflow import Run, S3
# from metaflow_extensions.deepspeed.plugins.azure_blob import AzureBlob
from metaflow.plugins.az_store import AzureBlob

from transformers import TrainerCallback


class DeepspeedHFTrainerS3Sync(TrainerCallback):

    def __init__(
        self,
        run_pathspec=None,
        s3_root=None,
        training_outputs_path="training_outputs",
        local_rank=None,
        node_index=None,
        push_from_all_nodes=False,  
    ):
        self.root = training_outputs_path
        self.run = Run(run_pathspec)
        self.s3_root = s3_root
        self.local_rank = local_rank
        self.node_index = node_index
        self.push_from_all_nodes = push_from_all_nodes

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Push the training outputs to Metaflow S3 on epoch end.
        Find other hooks at https://huggingface.co/docs/transformers/v4.38.1/en/main_classes/callback#transformers.TrainerCallback

        NOTE: As models get larger, be careful the amount of checkpoints you push to S3. You may, for example, only want to push the best checkpoint.
        """
        if os.path.exists(self.root) and (
            self.local_rank is not None and self.local_rank == 0
        ):
            try:
                if self.push_from_all_nodes or self.node_index == 0:
                    print(f"Pushing {self.root} to S3 on node {self.node_index}.")
                    tar_bytes = self._get_tar_bytes()
                    self._upload_to_s3(tar_bytes)
                    print(f"Pushed to {self.s3_path} on node {self.node_index}.")
            except Exception as e:
                print(f"Error pushing to S3 on node {self.node_index}: {e}")

    def _get_tar_bytes(self):
        "Zip from the root of the directory and return the bytes of the tar file."
        with tarfile.open(f"{self.root}.tar.gz", "w:gz") as tar:
            tar.add(self.root, arcname=os.path.basename(self.root))
        with open(f"{self.root}.tar.gz", "rb") as f:
            tar_bytes = f.read()
        return tar_bytes

    def _get_s3_client(self):
        "Return an S3 object based on the run or s3_root."
        if self.run:
            return S3(run=self.run)
        elif self.s3_root:
            return S3(s3root=self.s3_root)
        else:
            return S3(s3root=os.path.join(DATATOOLS_S3ROOT, self.root))

    def _upload_to_s3(self, tar_bytes):
        "Push the tar file to S3."
        s3 = self._get_s3_client()
        if s3 is None:
            return None
        if self.local_rank is not None and self.local_rank == 0:
            self.s3_path = s3.put(
                f"{self.root}-node-{self.node_index}.tar.gz", tar_bytes
            )
        else:
            self.s3_path = s3.put(f"{self.root}.tar.gz", tar_bytes)
        s3.close()

    def _download_from_s3(
        self, all_nodes: bool = False
    ) -> Union[bytes, Dict[str, bytes]]:
        "Pull the tar file(s) from S3."
        s3 = self._get_s3_client()
        candidate_paths = s3.list_paths()
        if all_nodes:
            tar_balls = {}
            for s3obj in candidate_paths:
                if self.root in s3obj.key:
                    obj = s3.get(s3obj.key)
                    tar_balls[obj.key] = obj.blob
            s3.close()
            return tar_balls
        elif self.node_index is not None:
            tar_bytes = s3.get(f"{self.root}-node-{self.node_index}.tar.gz").blob
        else:
            tar_bytes = s3.get(f"{self.root}.tar.gz").blob
        s3.close()
        return tar_bytes

    def _extract_tar(self, tar_bytes, path=None):
        """
        Extract the tar file to the root of the directory.
        If `path` is specified, assumed to be a file path and extract to that location.
        The use case for path is when downloading all checkpoints from many nodes nodes.
        """
        if path:
            with open(path, "wb") as f:
                f.write(tar_bytes)
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(path=path.replace(".tar.gz", ""))
            os.remove(path)
        else:
            with open(f"{self.root}.tar.gz", "wb") as f:
                f.write(tar_bytes)
            with tarfile.open(f"{self.root}.tar.gz", "r:gz") as tar:
                tar.extractall(path=os.path.dirname(self.root))
            os.remove(f"{self.root}.tar.gz")

    def download(self, all_nodes=False):
        if all_nodes:
            tar_balls = self._download_from_s3(all_nodes=all_nodes)
            for _path, _bytes in tar_balls.items():
                self._extract_tar(_bytes, path=_path)
        else:
            tar_bytes = self._download_from_s3()
            self._extract_tar(tar_bytes)


class DeepspeedHFTrainerAzureBlobSync(TrainerCallback):

    def __init__(
        self,
        run_pathspec=None,
        training_outputs_path="training_outputs",
        local_rank=None,
        node_index=None,
        push_from_all_nodes=False
    ):
        self.root = training_outputs_path
        self.run_pathspec = run_pathspec
        self.local_rank = local_rank
        self.node_index = node_index
        self.push_from_all_nodes = push_from_all_nodes

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Push the training outputs to Metaflow Azure Blob Storage on epoch end.
        Find other hooks at https://huggingface.co/docs/transformers/v4.38.1/en/main_classes/callback#transformers.TrainerCallback

        NOTE: As models get larger, be careful the amount of checkpoints you push to Azure Blob. You may, for example, only want to push the best checkpoint.
        """
        if os.path.exists(self.root) and (
            self.local_rank is not None and self.local_rank == 0
        ):
            try:
                if self.push_from_all_nodes or self.node_index == 0:
                    print(f'Zipping and pushing {self.root} to Azure Blob.')
                    tar_bytes = self._get_tar_bytes()
                    self._upload_to_azure_blob(tar_bytes)
                    print(f"Pushed to Azure Blob.")
            except Exception as e:
                print(f"Error pushing to Azure Blob: {e}")

    def _get_tar_bytes(self):
        "Zip from the root of the directory and return the bytes of the tar file."
        with tarfile.open(f"{self.root}.tar.gz", "w:gz") as tar:
            tar.add(self.root, arcname=os.path.basename(self.root))
        with open(f"{self.root}.tar.gz", "rb") as f:
            tar_bytes = f.read()
        return tar_bytes

    def _get_blob_store(self):
        "Return an Azure Blob object based on the run."
        return AzureBlob(run_pathspec=self.run_pathspec)

    def _upload_to_azure_blob(self, tar_bytes):
        "Push the tar file to Azure Blob."
        blob_store = self._get_blob_store()
        if self.local_rank is not None and self.local_rank == 0:
            blob_store.put(
                f"{self.root}-node-{self.node_index}.tar.gz", tar_bytes
            )
        else:
            blob_store.put(f"{self.root}.tar.gz", tar_bytes)

    def _download_from_azure_blob(
        self, all_nodes: bool = False
    ) -> Union[bytes, Dict[str, bytes]]:
        "Pull the tar file(s) from Azure Blob."
        blob_store = self._get_blob_store()
        candidate_paths = blob_store.list_paths()
        if all_nodes:
            tar_balls = {}
            for obj in candidate_paths:
                if self.root in obj.key:
                    obj = blob_store.get(obj.key)
                    tar_balls[obj.key] = obj.blob
            return tar_balls
        elif self.node_index is not None:
            tar_bytes = blob_store.get(f"{self.root}-node-{self.node_index}.tar.gz").blob
        else:
            tar_bytes = blob_store.get(f"{self.root}.tar.gz").blob_bytes
        return tar_bytes

    def _extract_tar(self, tar_bytes, path=None):
        """
        Extract the tar file to the root of the directory.
        If `path` is specified, assumed to be a file path and extract to that location.
        The use case for path is when downloading all checkpoints from many nodes nodes.
        """
        if path:
            with open(path, "wb") as f:
                f.write(tar_bytes)
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(path=path.replace(".tar.gz", ""))
            os.remove(path)
        else:
            with open(f"{self.root}.tar.gz", "wb") as f:
                f.write(tar_bytes)
            with tarfile.open(f"{self.root}.tar.gz", "r:gz") as tar:
                tar.extractall(path=os.path.dirname(self.root))
            os.remove(f"{self.root}.tar.gz")

    def download(self, all_nodes=False):
        if all_nodes:
            tar_balls = self._download_from_azure_blob(all_nodes=all_nodes)
            for _path, _bytes in tar_balls.items():
                self._extract_tar(_bytes, path=_path)
        else:
            tar_bytes = self._download_from_azure_blob()
            self._extract_tar(tar_bytes)