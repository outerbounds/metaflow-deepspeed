from metaflow import FlowSpec, step, deepspeed, kubernetes, current, environment
# from metaflow.profilers import gpu_profile

N_NODES = 1
IMAGE = "docker.io/eddieob/deepspeed:6"
MEMORY = "16000"
N_CPU = 2
N_GPU = 1


class MetaflowDeepspeedHFCallbackExample(FlowSpec):

    checkpoint_dir = "training_outputs"

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    # @gpu_profile(interval=1)
    @environment(vars={'NCCL_SOCKET_IFNAME': 'eth0'})
    @kubernetes(image=IMAGE, memory=MEMORY, cpu=N_CPU, gpu=N_GPU)
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

    @kubernetes(image=IMAGE, memory=MEMORY, cpu=N_CPU)
    @step
    def end(self):

        # Download the results from the S3 bucket.
        from metaflow.plugins.hf_callbacks import DeepspeedHFTrainerS3Sync
        checkpoint_handler = DeepspeedHFTrainerS3Sync(run_pathspec=f"{current.flow_name}/{current.run_id}")
        checkpoint_handler.download(all_nodes=True)

        # Print the results.
        import os
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if self.checkpoint_dir in os.path.join(root, file):
                    print(os.path.join(root, file))


if __name__ == "__main__":
    MetaflowDeepspeedHFCallbackExample()
