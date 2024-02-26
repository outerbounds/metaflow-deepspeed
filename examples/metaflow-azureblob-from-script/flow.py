from metaflow import FlowSpec, step, deepspeed, kubernetes, current
import json

N_NODES = 2
# IMAGE = "public.ecr.aws/p7g1e3j4/deepspeed:6"
IMAGE = "docker.io/eddieob/deepspeed:6"
MEMORY = "16000"
N_CPU = 2


class MetaflowDeepspeedAzureClientExample(FlowSpec):

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @kubernetes(image=IMAGE, memory="16000", cpu=N_CPU)
    @deepspeed
    @step
    def train(self):
        current.deepspeed.run(
            entrypoint="train.py",
            entrypoint_args={"run-pathspec": f"{current.flow_name}/{current.run_id}"},
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        from az_store import AzureBlob 
        blob_store = AzureBlob(run_pathspec=f"{current.flow_name}/{current.run_id}")
        for key_id in range(N_NODES * N_CPU):
            print(f"output_{key_id}.txt: ", blob_store.get(f"output_{key_id}.txt").text)


if __name__ == "__main__":
    MetaflowDeepspeedAzureClientExample()
