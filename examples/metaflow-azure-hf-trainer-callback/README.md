Run a Deepspeed process that uses a Huggingface trainer, and pushes checkpoints using a callback built into the `metaflow-deepspeed` package. 
Checkout `train.py` to see the script passed to the Deepspeed launcher.
You can find the experimental Huggingface callback implementation [here](../../metaflow_extensions/deepspeed/plugins/hf_callbacks.py), and are welcome to customize it to your needs!

```
python flow.py run
```