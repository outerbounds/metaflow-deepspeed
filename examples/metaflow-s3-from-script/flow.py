from metaflow import FlowSpec, step, deepspeed, kubernetes, current, S3
import json

N_NODES = 2
IMAGE = "docker.io/eddieob/deepspeed:6"
MEMORY = "16000"
N_CPU = 2


class MetaflowDeepspeedS3ClientExample(FlowSpec):

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @kubernetes(image=IMAGE, memory="16000", cpu=N_CPU)
    @deepspeed
    @step
    def train(self):
        current.deepspeed.run(
            entrypoint="train.py",
            entrypoint_args={"run_id": current.run_id, "flow_name": current.flow_name},
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        # Download the results of the @deepspeed @kubernetes step from the S3 bucket in a separate task running locally.
        with S3(run=self) as s3:
            for obj in s3.get_recursive(
                keys=[f"output_{i}.txt" for i in range(N_NODES * N_CPU)]
            ):
                print(obj.blob.decode("utf-8"))


if __name__ == "__main__":
    MetaflowDeepspeedS3ClientExample()
