from metaflow import FlowSpec, step, deepspeed, kubernetes, current, S3
import json

N_NODES = 2
IMAGE = "eddieob/deepspeed:6"
MEMORY = "16000"
N_CPU = 2


class MetaflowDeepspeedS3ClientExample(FlowSpec):

    checkpoint_dir = "training_outputs"

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @kubernetes(image=IMAGE, memory="16000", cpu=N_CPU)
    @deepspeed
    @step
    def train(self):

        # Run the deepspeed task.
        # In order to use Metaflow S3 client, we pass run-id and flow-name to the entrypoint.
        # This is necessary because deepspeed launches the entrypoint as a subprocess, 
        # so the script does not have access to the Metaflow current object to pull these values.
        current.deepspeed.run(
            entrypoint="train.py",
            entrypoint_args={
                "run-id": current.run_id,
                "flow-name": current.flow_name,
                "checkpoint-dir": self.checkpoint_dir,
            },
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @kubernetes(image=IMAGE, memory="16000", cpu=N_CPU)
    @step
    def end(self):

        # Download the results from the S3 bucket.
        from train import MetaflowS3Sync
        s3 = MetaflowS3Sync(run_pathspec=f"{current.flow_name}/{current.run_id}")
        s3.download(all_nodes=True)

        # Print the results.
        import os
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if self.checkpoint_dir in os.path.join(root, file):
                    print(os.path.join(root, file))


if __name__ == "__main__":
    MetaflowDeepspeedS3ClientExample()