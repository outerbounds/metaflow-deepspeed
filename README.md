### Introduction
[Deepspeed](https://www.deepspeed.ai/) is a highly scalable framework from Microsoft for distributed training and model serving. The Metaflow `@deepspeed` decorator helps you run these workflows inside of Metaflow tasks. 

### Features
- **Automatic SSH configuration**: [Multi-node Deepspeed jobs are built around OpenMPI or Horovod](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node). Like Metaflow's [`@mpi` decorator](https://github.com/outerbounds/metaflow-mpi), the `@deepspeed` decorator automatically configures SSH requirements between nodes, so you can focus on research code.
- **Seamless Python interface**: Metaflow's `@deepspeed` exposes a method `current.deepspeed.run` to make it easy to run Deepspeed commands on your transient MPI cluster, in the same way you'd launch Deepspeed from the terminal independent of Metaflow. A major design goal is to get the orchestration and other benefits of Metaflow, without requiring modification to research code.

### Installation
Install this experimental module:
```
pip install metaflow-deepspeed
```

### Getting Started
After installing the module, you can import the `deepspeed` decorator and use it in your Metaflow steps.
This exposes the `current.deepspeed.run` method, which you can map your terminal commands for running Deepspeed. 

```python
from metaflow import FlowSpec, step, deepspeed, current, batch, environment

class HelloDeepspeed(FlowSpec):

    @step
    def start(self):
        self.next(self.train, num_parallel=2)

    @environment(vars={"NCCL_SOCKET_IFNAME": "eth0"})
    @batch(gpu=8, cpu=64, memory=256000)
    @deepspeed
    @step
    def train(self):
        current.deepspeed.run(
            entrypoint="my_torch_dist_script.py"
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass
        
if __name__ == "__main__":
    HelloDeepspeed()
```

### Examples
| Directory | MPI program description |
| :--- | ---: |
| [Hello Deepspeed](examples/hello/README.md) | The easiest way to understand deepspeed and get hands on. |  
| [BERT](examples/bert/README.md) | Train your BERT model using Deepspeed! |  
| [Dolly](examples/dolly/README.md) | A multi-node implementation of [Databricks' Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm). |  

### License
`metaflow-deepspeed` is distributed under the <u>Apache License</u>.