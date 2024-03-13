from metaflow import FlowSpec, step, deepspeed, current, kubernetes, card
from metaflow.cards import Markdown
import json

N_NODES = 2
# IMAGE = "public.ecr.aws/p7g1e3j4/deepspeed:6"
IMAGE = "docker.io/eddieob/deepspeed:6"
MEMORY = "16000"
N_CPU = 2

class CPUOnlyDeepspeedDistributedTest(FlowSpec):

    @card
    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @kubernetes(image=IMAGE, memory=MEMORY, cpu=N_CPU)
    @deepspeed
    @step
    def train(self):
        current.deepspeed.run(entrypoint="train.py")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    CPUOnlyDeepspeedDistributedTest()
