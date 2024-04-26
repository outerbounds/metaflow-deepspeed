Uses a Huggingface trainer that publishes cards that track training progress and metrics in real-time. Checkout `train.py` to see the script passed to the Deepspeed launcher. You can find the experimental Huggingface callback implementation [here](../../metaflow_extensions/deepspeed/plugins/deepspeed_libs/hugging_face/card_callback.py), and are welcome to customize it to your needs! The [flow](./flow.py) file uses the `@huggingface_card` callback to track training progress and metrics in real-time.

```
python flow.py run
```