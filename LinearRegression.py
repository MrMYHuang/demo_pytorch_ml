import torch
from torch import Tensor
from torch.nn import Module, Parameter, L1Loss

class LinearRegression(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(torch.rand(1))
        self.bias = Parameter(torch.rand(1))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight + self.bias