import deepspeed
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
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.criterion = torch.nn.MSELoss(reduction="sum")

    def forward(self, batch):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.

        NOTE: A gotcha about deepspeed is the forward function HAS TO return the loss.
              This deviates from most MySubclass(nn.Module).forward funcs on the internet which typically return logits.
        """
        x, y = batch[0].cuda(), batch[1].cuda()
        y_pred = self.a + self.b * x + self.c * x**2 + self.d * x**3
        loss = self.criterion(y_pred, y)
        return loss

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f"y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3"


def init_backend(model):
    "Configure setup required for distributed data parallel and mixed precision training"
    return deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json",
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
    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # Construct PyTorch dataset and corresponding dataloader.
    train_data = data_utils.TensorDataset(x, y)
    train_loader = data_utils.DataLoader(train_data, batch_size=1, shuffle=True)

    # Construct model by instantiating the class defined above.
    model = Polynomial3()

    # Initialize deepspeed runtime environment.
    model_engine, optimizer, _, _ = init_backend(model)

    # Training loop, with distributed training orchestrated by deepspeed.
    train(model_engine, train_loader)


if __name__ == "__main__":
    main()