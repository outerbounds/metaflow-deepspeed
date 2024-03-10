from metaflow import FlowSpec, step, deepspeed, current, kubernetes, environment, S3

N_NODES = 2
# IMAGE = "public.ecr.aws/p7g1e3j4/deepspeed:6"
IMAGE = "docker.io/eddieob/deepspeed:6"
MEMORY = "16000"
N_CPU = 2


class MetaflowDeepspeedDirectoryUpload(FlowSpec):

    @step
    def start(self):
        self.next(self.train, num_parallel=N_NODES)

    @kubernetes(image=IMAGE, memory=MEMORY, cpu=N_CPU, use_tmpfs=True)
    @deepspeed
    @step
    def train(self):

        # Change this to the local path you want to use.
        self.local_output_dir = f"{current.tempdir}/results_from_tmpfs_storage"

        # Change this to the azure blob location you want to sync the local path with.
        self.output_dir = "results_from_tmpfs_storage"

        # Run the deepspeed task with the local and azure blob output directories.
        # notice the entrypoint args takes in the local output directory that is synced,
        # To simulate e.g., a model training process that dumps checkpoints there.
        # If you don't want to automatically push to Azure Blob, don't specify push_results_dir_to_cloud, local_output_dir, or cloud_output_dir.
        current.deepspeed.run(
            entrypoint="train.py",
            entrypoint_args={"output-dir": self.local_output_dir},
            local_output_dir=self.local_output_dir,
            cloud_output_dir=self.output_dir,  # If you don't specify this, it will be metaflow_temp, or whatever you change tmpfs path to.
            push_results_dir_to_cloud=True,
        )

        self.next(self.join)

    @step
    def join(self, inputs):
        self.output_dir = inputs[0].output_dir
        self.next(self.end)

    @step
    def end(self):
        """
        Read the result from another computer to verify that the file was written to object storage.
        This is a separate task running locally.
        Follow the instructions that the current.deepspeed.run prints to stdout when push_results_dir_to_cloud=True.
        """
        from metaflow.plugins.az_store import AzureBlob
        from metaflow.plugins.deepspeed_libs.constants import DEEPSPEED_SUFFIX
        blob_store = AzureBlob(run_pathspec=f"{DEEPSPEED_SUFFIX}/{current.flow_name}/{current.run_id}", step_name='train')
        paths = blob_store.list_paths([self.output_dir])
        blob_store.get_files([(p.key, p.key) for p in paths])
        for p in paths:
            with open(p.key, "r") as f:
                print(f.read())

if __name__ == "__main__":
    MetaflowDeepspeedDirectoryUpload()
