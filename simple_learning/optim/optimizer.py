from typing import Collection

import numpy as np

from simple_learning.nn.parameter import Parameter


class Optimizer:
    def __init__(self, parameters: Collection[Parameter]):
        self._parameters = parameters

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self._parameters:
            param.grad = np.zeros_like(param.grad)


