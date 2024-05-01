from argparse import ArgumentParser
import time
from deepspeed import comm as dist
from deepspeed import init_distributed

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    args = parser.parse_args()
    init_distributed(dist_backend="gloo")
    global_rank = dist.get_rank()
    time.sleep(3)
    dist.barrier()
    print(f"I am global rank {global_rank} | local rank {args.local_rank}")
    time.sleep(3)
