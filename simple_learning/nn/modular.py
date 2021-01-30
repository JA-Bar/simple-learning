import numpy as np

import simple_learning.grad.functional as F

from .module import Module
from .parameter import Parameter


__all__ = ['Linear']


class Linear(Module):
    """Apply an affine transformation (bias=True) or linear transformation (bias=False)
    to a some input, where the weight matrix has its values initialized by the Uniform He method.

    Args:
        in_features: size of the input features.
        out_features: size of the output features.
        bias: wether to add a bias weight or not.

    Shapes:
        input: (N, I) where N are the number of samples and I the input features.
        output: (N, O) where O are the output features.
    """
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


