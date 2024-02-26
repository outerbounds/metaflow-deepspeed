import deepspeed
import argparse
import torch.distributed as dist
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.datasets
import torch
import torch.nn as nn

# -*- coding: utf-8 -*-
import torch
import math


class Polynomial3(torch.nn.Module):
    def __init__(self, device):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(())).to(device)
        self.b = torch.nn.Parameter(torch.randn(())).to(device)
        self.c = torch.nn.Parameter(torch.randn(())).to(device)
        self.d = torch.nn.Parameter(torch.randn(())).to(device)
        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.device = device

    def forward(self, batch):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.

        NOTE: A gotcha about deepspeed is the forward function HAS TO return the loss.
              This deviates from most MySubclass(nn.Module).forward funcs on the internet which typically return logits.
        """
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        y_pred = self.a + self.b * x + self.c * x**2 + self.d * x**3
        loss = self.criterion(y_pred, y)
        return loss

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f"y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3"


def init_backend(model, backend="nccl"):
    "Configure setup required for distributed data parallel and mixed precision training"
    deepspeed.init_distributed(dist_backend=backend)
    return deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json",
        dist_init_required=False
    )


def train(model_engine, data_loader):
    "Distributed data parallel training, in mixed precision, with a pre-defined learning rate scheduler"

    for step, batch in enumerate(data_loader):
        # forward() method handles scaling the loss to avoid precision loss in the gradients
        loss = model_engine(batch)

        # runs backpropagation with gradients averaged across data parallel processes
        model_engine.backward(loss)

        # weight update
        model_engine.step()

        print("step: %s, loss: %s" % (step, loss.item()))


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "gloo" if device.type == "cpu" else "nccl"

    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000).to(device)
    y = torch.sin(x).to(device)

    # Construct PyTorch dataset and corresponding dataloader.
    train_data = data_utils.TensorDataset(x, y)
    train_loader = data_utils.DataLoader(train_data, batch_size=1, shuffle=True)

    # Construct model by instantiating the class defined above.
    model = Polynomial3(device)

    # Initialize deepspeed runtime environment.
    model_engine, optimizer, _, _ = init_backend(model, backend)

    # Training loop, with distributed training orchestrated by deepspeed.
    train(model_engine, train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    args = parser.parse_args()
    main()
