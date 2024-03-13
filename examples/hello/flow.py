from metaflow import FlowSpec, step, deepspeed, current, kubernetes, environment
import json
from metaflow.profilers import gpu_profile

N_NODES = 4
IMAGE = "docker.io/eddieob/deepspeed:6"
N_GPU = 1
MEMORY = "12000"
N_CPU = 2

class HelloDeepspeed(FlowSpec):

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @environment(vars={"NCCL_SOCKET_IFNAME": "eth0"})
    @kubernetes(image=IMAGE, gpu=N_GPU, memory=MEMORY, cpu=N_CPU)
    @deepspeed
    @gpu_profile(interval=1)
    @step
    def train(self):
        current.deepspeed.run(
            entrypoint="hi-deepspeed.py",
            deepspeed_args={
                # "bind_cores_to_rank": "",
                # "launcher": "impi"
            }
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    HelloDeepspeed()
