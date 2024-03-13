from metaflow import FlowSpec, step, deepspeed, current, batch, environment, S3

N_NODES = 2
IMAGE = "public.ecr.aws/p7g1e3j4/deepspeed:6" #"eddieob/deepspeed:6"
MEMORY = "16000"
N_CPU = 2


class MetaflowDeepspeedDirectoryUpload(FlowSpec):

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @batch(image=IMAGE, memory=MEMORY, cpu=N_CPU, use_tmpfs=True)
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
            entrypoint_args={"output-dir": self.local_output_dir},
            local_output_dir=self.local_output_dir,
            cloud_output_dir=self.s3_output_dir,  # If you don't specify this, it will be metaflow_temp, or whatever you change tmpfs path to.
            push_results_dir_to_cloud=True,
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
        from metaflow.metaflow_config import DATASTORE_SYSROOT_S3
        from metaflow.plugins.deepspeed_libs.constants import DEEPSPEED_SUFFIX
        import os
        s3root = os.path.join(DATASTORE_SYSROOT_S3, DEEPSPEED_SUFFIX, current.flow_name, current.run_id, 'train')
        with S3(s3root=s3root) as s3:
            objs = s3.get_recursive(keys=[self.s3_output_dir])
            for obj in objs:
                with open(obj.path, "r") as f:
                    result = f.read().strip()
                    print(result)


if __name__ == "__main__":
    MetaflowDeepspeedDirectoryUpload()
