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
        executor = DeepspeedExecutor.for_single_node(
            flow=self, 
            use_gpu=False, 
            n_slots = N_CPU
        )
        executor.run(entrypoint="train.py")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    SingleNodeDeepspeedTest()
