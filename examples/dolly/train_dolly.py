"""
This flow wraps the training code for Databricks Dolly model. 
You can find the original source code here: https://github.com/databrickslabs/dolly

The extensions this flow has include:
    - Easy to package to run on AWS Batch or Kubernetes
    - GPU profiling
    - A small streamlit app to interact with the model after training in app.py
"""

from metaflow import FlowSpec, step, Parameter, environment, S3, deepspeed, pypi, kubernetes, current
from metaflow import IncludeFile
import subprocess
from my_decorators import gpu_profile
from consts import *

def print_volumes():
    print("Volumes:")
    subprocess.run(["df", "-h"])
    print("")
    print("Mounted volumes:")
    subprocess.run(["mount"])
    print("")

class TrainDolly(FlowSpec):

    # config
    training_output_dir = Parameter(name="training_output_dir", default="training_output", help="Directory to store training output.")
    deepspeed_config = Parameter(name="config", default="ds_config.json", help="DeepSpeed config file.") 
    # TODO: The deepspeed config file could use IncludeFile. It is simple, but a bit clunky to use Parameter + --package-suffixes.

    # hyperparameters
    learning_rate = Parameter("learning_rate", default="1e-5", help="Learning rate for training.")
    batch_size = Parameter("batch_size", default="16", help="Batch size for training & evaluation.")

    debug_log = Parameter("debug_log", is_flag=True, default=False, help="Whether to log debug messages.")
    sample_percentage = Parameter("sample-percentage", default=None, help="Percentage of data to use for training.")
    push_model_to_s3 = Parameter("push-model-to-s3", default=True, help="Whether to push model to S3 after training.")

    @step
    def start(self):
        print(self.deepspeed_config)
        self.next(self.train, num_parallel=N_NODES)

    @deepspeed
    @kubernetes(image=IMAGE, cpu=N_CPU, gpu=N_GPU, memory=MEMORY)
    @environment(vars = {"TOKENIZERS_PARALLELISM": "false", "NCCL_SOCKET_IFNAME": "eth0"})
    @gpu_profile(interval=1)
    @step
    def train(self):

        import os 
        import subprocess
        from datetime import datetime
        import tempfile
        import json

        # Configure local output directory.
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        model_name = "dolly"
        checkpoint_dir_name = f"{model_name}__{timestamp}"
        root_path = os.getcwd()
        local_training_root = os.path.join(os.getcwd(), self.training_output_dir)
        os.makedirs(local_training_root, exist_ok=True)
        self.local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)

        # Use the @deepspeed decorator
        current.deepspeed.run(
            deepspeed_args=[
                "--num_nodes=%d" % N_NODES,
                "--num_gpus=%d" % N_GPU,
            ],
            entrypoint="trainer.py", 
            entrypoint_args=[
                "--deepspeed", self.deepspeed_config,
                "--epochs", "1",
                "--local-output-dir", self.local_output_dir,
                "--per-device-train-batch-size", self.batch_size,
                "--per-device-eval-batch-size", self.batch_size,
                "--lr", self.learning_rate
            ] + (["--debug-log"] if self.debug_log else []) + (["--sample-percentage", self.sample_percentage] if self.sample_percentage else [])
        )
        print("\nDeepspeed process completed!")

        # Put all files in the local_output_directory in S3 bucket, versioned with this flow run.
        if self.push_model_to_s3:
            print("Writing model to s3...\n\n")
            self.s3_output_dir = os.path.join(self.training_output_dir, checkpoint_dir_name)
            with S3(run=self) as s3:
                self.filepath_tuples = [
                    (f"{self.s3_output_dir}/{f}", os.path.join(self.local_output_dir, f)) 
                    for f in os.listdir(self.local_output_dir) 
                    if os.path.isfile(os.path.join(self.local_output_dir, f))
                ]
                print(s3.put_files(self.filepath_tuples))

            # This conditional block enables resuming from this checkpoint in a subsequent flow or a notebook:
            print("\n\nModel written to s3!\n\n")
            print("Access it with:\n\n")
            print("""
                from metaflow import Flow, S3
                import transformers
                run = Flow('%s').latest_run
                local_output_path = run.data.local_output_path
                S3(run=flow.run) as s3:
                for local_path, s3_path in run.data.filepath_tuples:
                    obj = s3.get(s3_path)
                    os.rename(obj.path, local_path)
                model = transformers.AutoModelForCausalLM.from_pretrained(local_output_path, ...)
            """ % current.flow_name)

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print(f"\n Woohoo! You've trained your own LLM! \U0001f389 \U0001f389 \U0001f389")

if __name__ == '__main__':
    TrainDolly()