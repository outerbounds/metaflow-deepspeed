from argparse import ArgumentParser
import time
from metaflow import S3, Run
from random import choice


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
