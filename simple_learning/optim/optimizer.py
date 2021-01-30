from typing import Collection

import numpy as np

from simple_learning.nn.parameter import Parameter


class Optimizer:
    """Interface for optimizers.

    All objects that modify the parameters of a Module should subclass Optimizer.

    A custom optimizer should implement two methods - __init__: where the __init__
    method of the base class Optimizer should be called, and step: where the way
    parameters are updated is defined.
    """
    def __init__(self, parameters: Collection[Parameter]):
        self._parameters = parameters

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self._parameters:
            param.grad = np.zeros_like(param.data)

