from metaflow import FlowSpec, step, kubernetes

IMAGE = "docker.io/eddieob/deepspeed:6"
MEMORY = "16000"
N_CPU = 2


class SingleNodeDeepspeedTest(FlowSpec):
    @step
    def start(self):
        self.next(self.train)

    @kubernetes(image=IMAGE, memory=MEMORY, cpu=N_CPU)
    @step
    def train(self):
        from metaflow.plugins.deepspeed_libs.executor import DeepspeedExecutor
        from metaflow.plugins.deepspeed_libs.mpi_setup import setup_mpi_env

        hosts = setup_mpi_env(
            flow_datastore=self._datastore.parent_datastore,
            n_slots=N_CPU,
        )
        exe = DeepspeedExecutor(
            hosts=hosts,
            is_gpu=False,  # default
            flow=self,
            flow_datastore=self._datastore.parent_datastore,
            # n_slots_per_host=N_GPU, # sets num_gpus arg. Alternatively, use deepspeed_args in exe.run.
        )
        exe.run(entrypoint="train.py")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SingleNodeDeepspeedTest()
