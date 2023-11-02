from metaflow import FlowSpec, step, deepspeed, current, kubernetes, batch, environment
import json

N_NODES = 2
IMAGE = "eddieob/deepspeed:bert-example"
N_GPU = 1
MEMORY = "32000"
N_CPU = 8


class CoreweaveBERT(FlowSpec):
    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @environment(vars={"NCCL_SOCKET_IFNAME": "eth0"})
    @kubernetes(image=IMAGE, gpu=N_GPU, memory=MEMORY, cpu=N_CPU)
    @deepspeed
    @step
    def train(self):
        current.deepspeed.run(
            # deepspeed_args=["--bind_cores_to_rank"],
            entrypoint="train.py",
            entrypoint_args=["--checkpoint_dir", "experiment_deepspeed"],
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CoreweaveBERT()
