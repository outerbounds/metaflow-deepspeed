import os
import time
import tarfile
import argparse
from random import choice
from typing import Union, Dict

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainerCallback

try: # in case you want to download to local env without deepspeed
    from deepspeed import comm as dist
    from deepspeed import init_distributed
except ImportError:
    pass

from az_store import AzureBlob


class MetaflowAzureBlobSync(TrainerCallback):

    def __init__(
        self,
        run_pathspec=None,
        training_outputs_path="training_outputs",
        local_rank=None,
        node_index=None,
    ):
        self.root = training_outputs_path
        self.run_pathspec = run_pathspec
        self.local_rank = local_rank
        self.node_index = node_index

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Push the training outputs to Metaflow S3 on epoch end.
        Find other hooks at https://huggingface.co/docs/transformers/v4.38.1/en/main_classes/callback#transformers.TrainerCallback

        NOTE: As models get larger, be careful the amount of checkpoints you push to Azure Blob. You may, for example, only want to push the best checkpoint.
        """
        if os.path.exists(self.root) and (
            self.local_rank is not None and self.local_rank == 0
        ):
            print(f'Zipping and pushing {self.root} to Azure Blob.')
            tar_bytes = self._get_tar_bytes()
            self._upload_to_azure_blob(tar_bytes)
            print(f"Pushed to Azure Blob.")

    def _get_tar_bytes(self):
        "Zip from the root of the directory and return the bytes of the tar file."
        with tarfile.open(f"{self.root}.tar.gz", "w:gz") as tar:
            tar.add(self.root, arcname=os.path.basename(self.root))
        with open(f"{self.root}.tar.gz", "rb") as f:
            tar_bytes = f.read()
        return tar_bytes

    def _get_blob_store(self):
        "Return an S3 object based on the run or s3_root."
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


def main(
    checkpoint_dir: str = "training_outputs",
    ds_name: str = "rotten_tomatoes",
    model_name: str = "distilbert-base-uncased",
    split: str = "train[:20%]",
    run_id: str = None,
    flow_name: str = None,
    local_rank: int = None,
    global_rank: int = None,
):
    dataset = load_dataset(ds_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenize_dataset = lambda dataset: tokenizer(dataset["text"])
    dataset = dataset.map(tokenize_dataset, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=10,
        warmup_steps=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            MetaflowAzureBlobSync(
                run_pathspec=f"{flow_name}/{run_id}",
                training_outputs_path=checkpoint_dir,
                local_rank=local_rank,
                node_index=global_rank // int(os.environ["MF_PARALLEL_NUM_NODES"]),
            )
        ],
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="training_outputs")
    parser.add_argument("--run-id", type=str, default=None, help="Metaflow run id")
    parser.add_argument(
        "--flow-name", type=str, default=None, help="Metaflow flow_name"
    )
    args = parser.parse_args()
    init_distributed(dist_backend="gloo")
    main(
        checkpoint_dir=args.checkpoint_dir,
        run_id=args.run_id,
        flow_name=args.flow_name,
        local_rank=args.local_rank,
        global_rank=dist.get_rank(),
    )
