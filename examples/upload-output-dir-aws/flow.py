from metaflow import FlowSpec, step, deepspeed, current, kubernetes, environment, S3
import json

N_NODES = 2
IMAGE = "public.ecr.aws/p7g1e3j4/deepspeed:6" #"eddieob/deepspeed:6"
MEMORY = "16000"
N_CPU = 2


class MetaflowDeepspeedDirectoryUpload(FlowSpec):

    messages = [
        # one of these messages will be written to a file in tmpfs by each process.
        "good evening friend",
        "bonsoir mon ami",
        "こんばんは、私の友人",
        "guten Abend mein Freund",
        "boa noite meu amigo",
    ]

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @kubernetes(image=IMAGE, memory=MEMORY, cpu=N_CPU, use_tmpfs=True)
    @deepspeed
    @step
    def train(self):
        from random import choice

        # Change this to the local path you want to use.
        self.local_output_dir = f"{current.tempdir}/results_from_tmpfs_storage"

        # Change this to the s3 location you want to sync the local path with.
        self.s3_output_dir = "results_from_tmpfs_storage"

        # Run the deepspeed task with the local and s3 output directories.
        # notice the entrypoint args takes in the local output directory that is synced,
        # To simulate e.g., a model training process that dumps checkpoints there.
        # If you don't want to automatically push to S3, don't specify push_results_dir_to_s3, local_output_dir, or s3_output_dir.
        current.deepspeed.run(
            entrypoint="train.py",
            entrypoint_args={
                "output-dir": self.local_output_dir,
                "contents": choice(self.messages),
            },
            local_output_dir=self.local_output_dir,
            s3_output_dir=self.s3_output_dir,  # If you don't specify this, it will be metaflow_temp, or whatever you change tmpfs path to.
            push_results_dir_to_s3=True,
        )

        self.next(self.join)

    @step
    def join(self, inputs):
        self.s3_output_dir = inputs[0].s3_output_dir
        self.next(self.end)

    @step
    def end(self):
        """
        Read the result from another computer to verify that the file was written to object storage.
        This is a separate task running locally.
        Follow the instructions that the current.deepspeed.run prints to stdout when push_results_dir_to_s3=True.
        """
        with S3(run=self) as s3:  # result versioned by this flow
            objs = s3.get_recursive(keys=[self.s3_output_dir])
            for obj in objs:
                with open(obj.path, "r") as f:
                    result = f.read().strip()
                    print(result)


if __name__ == "__main__":
    MetaflowDeepspeedDirectoryUpload()
