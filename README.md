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
| [CPU Check](examples/cpu-only-check/README.md) | The easiest way to check your Deepspeed infrastructure on CPUs. |
| [Hello Deepspeed](examples/hello/README.md) | The easiest way to check your Deepspeed infrastructure on GPUs. |  
| [BERT](examples/bert/README.md) | Train your BERT model using Deepspeed! |  
| [Dolly](examples/dolly/README.md) | A multi-node implementation of [Databricks' Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm). |  

#### Cloud-specific use cases
| Directory | MPI program description |
| :--- | ---: |
| [Automatically upload a directory on AWS](examples/upload-output-dir-aws/README.md) | Push a checkpoint of any directory to S3 after the Deepspeed process completes. |
| [Automatically upload a directory on Azure](examples/upload-output-dir-azure/README.md) | Push a checkpoint of any directory to Azure Blob storage after the Deepspeed process completes. |
| [Use Metaflow S3 client from the Deepspeed process](examples/metaflow-s3-from-script/README.md) | Upload arbitrary bytes to S3 storage from the Deepspeed process. |  
| [Use Metaflow Azure Blob client from the Deepspeed process](examples/metaflow-azureblob-from-script/README.md) | Upload arbitrary bytes to Azure Blob storage from the Deepspeed process. |  
| [Use a Metaflow Huggingface checkpoint on S3](examples/metaflow-s3-hf-trainer-callback/README.md) | Push a checkpoint to S3 at the end of each epoch using a customizable Huggingface callback. See the implementation [here](./metaflow_extensions/deepspeed/plugins/hf_callbacks.py) to build your own. |  
| [Use a Metaflow Huggingface checkpoint on Azure](examples/metaflow-azure-hf-trainer-callback/README.md) | Push a checkpoint to Azure Blob storage at the end of each epoch using a customizable Huggingface callback. See the implementation [here](./metaflow_extensions/deepspeed/plugins/hf_callbacks.py) to build your own. |  


### License
`metaflow-deepspeed` is distributed under the <u>Apache License</u>.