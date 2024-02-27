from argparse import ArgumentParser
import time
import os
from deepspeed import comm as dist
from deepspeed import init_distributed

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/tmp")
    parser.add_argument("--contents", type=str, default="hi!")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    args = parser.parse_args()

    # simulate some process doing work that writes outputs to the directory
    init_distributed(dist_backend="gloo")
    global_rank = dist.get_rank()
    time.sleep(3)
    if not os.path.exists(args.output_dir) and args.local_rank == 0:
        os.makedirs(args.output_dir)
    dist.barrier()
    with open(f"{args.output_dir}/output_{global_rank}.txt", "w") as f:
        f.write(f"Process {global_rank}: {args.contents}")
    time.sleep(3)
