import os
import time
from random import choice
from argparse import ArgumentParser

# Experimental version of Azure Blob support.
from az_store import AzureBlob

def do_metaflow_operation(run_pathspec, global_rank):

    # business logic.
    animal = choice(["hippopotamus", "giraffe", "monkey", "lion", "tiger", "bear"])

    # push results to object store.
    datastore = AzureBlob(run_pathspec=run_pathspec)
    datastore.put(
        f"output_{global_rank}.txt",
        f"Process {global_rank} wants a {animal} for Christmas.",
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument("--run-pathspec", type=str, default=None, help="Metaflow flow id and run id")
    args = parser.parse_args()

    time.sleep(2)
    from deepspeed import comm as dist
    from deepspeed import init_distributed

    init_distributed(dist_backend="gloo")
    do_metaflow_operation(
        run_pathspec=args.run_pathspec, global_rank=dist.get_rank()
    )
    time.sleep(2)
