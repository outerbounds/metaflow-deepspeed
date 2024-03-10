import os
import time
import tarfile
import argparse
from random import choice
from typing import Union, Dict

from metaflow.plugins.hf_callbacks import DeepspeedHFTrainerAzureBlobSync

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

from metaflow.plugins.az_store import AzureBlob

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
        
        # save 2 checkpoints (best one and last one)
        save_total_limit = 2,
        save_steps=10,
        save_strategy = "steps", 
        load_best_model_at_end=False,
        save_on_each_node=False, # default. Notice relation to DeepspeedHFTrainerS3Sync.push_from_all_nodes.

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
            DeepspeedHFTrainerAzureBlobSync(
                run_pathspec=f"{flow_name}/{run_id}",
                training_outputs_path=checkpoint_dir,
                local_rank=local_rank,
                node_index= (1 + global_rank) // int(os.environ['MF_PARALLEL_NUM_NODES']), # NOTE: cannot use MF_PARALLEL_NODE_INDEX inside subprocess
                push_from_all_nodes=False, # default. Notice relation to TrainingArguments.save_on_each_node.
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
    init_distributed(dist_backend="nccl")
    main(
        checkpoint_dir=args.checkpoint_dir,
        run_id=args.run_id,
        flow_name=args.flow_name,
        local_rank=args.local_rank,
        global_rank=dist.get_rank(),
    )
