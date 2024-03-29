from metaflow import FlowSpec, step, deepspeed, kubernetes, current
from metaflow.profilers import gpu_profile

N_NODES = 2
IMAGE = "docker.io/eddieob/deepspeed:6"
MEMORY = "16000"
N_GPU = 1
N_CPU = 2


class MetaflowDeepspeedHFCallbackExample(FlowSpec):

    checkpoint_dir = "training_outputs"

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @gpu_profile(interval=1)
    @kubernetes(image=IMAGE, memory=MEMORY, cpu=N_CPU, gpu=N_GPU)
    @deepspeed
    @step
    def train(self):

        # Run the deepspeed task.
        # In order to use the Azure Blob client, we pass run-id and flow-name to the entrypoint.
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

        # Download the results from the Azure Blob storage bucket.
        from metaflow.plugins.hf_callbacks import DeepspeedHFTrainerAzureBlobSync
        blob_store = DeepspeedHFTrainerAzureBlobSync(run_pathspec=f"{current.flow_name}/{current.run_id}")
        blob_store.download(all_nodes=True)
        # You could use the above to download a pretrained checkpoint in the `train` step, instead of training from scratch.

        # Print the results, showing that the files were downloaded from the Azure Blob storage bucket.
        import os
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if self.checkpoint_dir in os.path.join(root, file):
                    print(os.path.join(root, file))


if __name__ == "__main__":
    MetaflowDeepspeedHFCallbackExample()
