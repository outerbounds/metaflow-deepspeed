from metaflow import FlowSpec, step, deepspeed, kubernetes, current
from metaflow.profilers import gpu_profile
from metaflow.plugins.hf_info_card import huggingface_card

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

    @huggingface_card
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

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MetaflowDeepspeedHFCallbackExample()
