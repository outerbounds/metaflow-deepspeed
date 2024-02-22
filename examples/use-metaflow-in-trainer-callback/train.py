import os
import time
import tarfile
import argparse
from random import choice
from typing import Union, Dict

from metaflow.metaflow_config import DATATOOLS_S3ROOT
from metaflow import S3, Run
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainerCallback
from deepspeed import comm as dist
from deepspeed import init_distributed


class MetaflowS3Sync(TrainerCallback):

    def __init__(
        self,
        run_pathspec=None,
        s3_root=None,
        training_outputs_path="training_outputs",
        local_rank=None,
        node_index=None,
    ):
        self.root = training_outputs_path
        self.run = Run(run_pathspec)
        self.s3_root = s3_root
        self.local_rank = local_rank
        self.node_index = node_index

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Push the training outputs to Metaflow S3 on epoch end.
        Find other hooks at https://huggingface.co/docs/transformers/v4.38.1/en/main_classes/callback#transformers.TrainerCallback

        NOTE: As models get larger, be careful the amount of checkpoints you push to S3. You may, for example, only want to push the best checkpoint.
        """
        if os.path.exists(self.root) and (
            self.local_rank is not None and self.local_rank == 0
        ):
            print(f"Pushing {self.root} to S3.")
            tar_bytes = self._get_tar_bytes()
            self._upload_to_s3(tar_bytes)
            print(f"Pushed to {self.s3_path}.")

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
            MetaflowS3Sync(
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
