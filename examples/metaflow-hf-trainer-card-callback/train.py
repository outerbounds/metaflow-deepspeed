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

from metaflow.huggingface_card_callback import MetaflowHuggingFaceCardCallback, MetaflowHuggingFaceProfilerCallback

def main(
    checkpoint_dir: str = "training_outputs",
    ds_name: str = "rotten_tomatoes",
    model_name: str = "distilbert-base-uncased",
    train_split: str = "train[:50%]",
    eval_split: str = "validation",
    run_id: str = None,
    flow_name: str = None,
    local_rank: int = None,
    global_rank: int = None,
):
    train_ds = load_dataset(ds_name, split=train_split)
    eval_ds = load_dataset(ds_name, split=eval_split)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenize_dataset = lambda dataset: tokenizer(dataset["text"])
    train_ds = train_ds.map(tokenize_dataset, batched=True)
    eval_ds = eval_ds.map(tokenize_dataset, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        logging_steps=1,
        # save 2 checkpoints (best one and last one)
        save_total_limit = 0,
        save_steps=200,
        save_strategy = "steps", 
        load_best_model_at_end=False,
        save_on_each_node=False, # default. Notice relation to DeepspeedHFTrainerS3Sync.push_from_all_nodes.

        warmup_steps=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            MetaflowHuggingFaceCardCallback(
                tracked_metrics = [
                    "loss",
                    "learning_rate",
                    "grad_norm",
                    "eval_loss",
                ]
            ),
            MetaflowHuggingFaceProfilerCallback(
                tracking_metrics= [
                    "cpu_memory_usage", 
                    "cuda_memory_usage",
                    "self_cpu_memory_usage",
                    "self_cuda_memory_usage"

                ]
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
