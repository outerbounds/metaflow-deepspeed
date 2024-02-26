from metaflow import FlowSpec, step, deepspeed, current, kubernetes, environment
import json

N_NODES = 4
IMAGE = "public.ecr.aws/p7g1e3j4/deepspeed:6"
N_GPU = 1
MEMORY = "16000"
N_CPU = 2

class HelloDeepspeed(FlowSpec):

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @environment(vars={"NCCL_SOCKET_IFNAME": "eth0"})
    @kubernetes(image=IMAGE, gpu=N_GPU, memory=MEMORY, cpu=N_CPU)
    @deepspeed
    @step
    def train(self):
        current.deepspeed.run(
            entrypoint="hi-deepspeed.py",
            entrypoint_args={
                "bind_cores_to_rank": "",
                "launcher": "impi"
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
