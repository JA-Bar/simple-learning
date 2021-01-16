import numpy as np

import simple_learning.grad.functional as F

from .module import Module
from .parameter import Parameter


__all__ = ['Linear']


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()

        # Uniform He initialization for weights
        std = np.sqrt(2) / np.sqrt(in_features)
        bound = np.sqrt(3) * std
        weight = np.random.uniform(-bound, bound, size=(out_features, in_features))
        self.weight = Parameter(weight)

        if bias:
            # Uniform He initialization for bias
            bound = 1 / np.sqrt(in_features)
            bias = np.random.uniform(-bound, bound, size=(1, out_features))
            self.bias = Parameter(bias)
        else:
            self.bias = Parameter.zeros((1, out_features), requires_grad=False)

    def forward(self, tensor):
        return F.Linear.apply(tensor, self.weight, self.bias)


