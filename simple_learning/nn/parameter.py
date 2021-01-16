from typing import Any

from simple_learning import Tensor


class Parameter(Tensor):
    def __init__(self, data: Any, requires_grad=True, parameter_group='all'):
        super().__init__(data, requires_grad=requires_grad)

        self.parameter_group = parameter_group
        self.buffer = 0

    # convenience wrappers to return Parameter
    @staticmethod
    def zeros(*args, **kwargs):
        return Parameter(Tensor.zeros(*args, **kwargs))

    @staticmethod
    def ones(*args, **kwargs):
        return Parameter(Tensor.randn(*args, **kwargs))

    @staticmethod
    def rand(*args, **kwargs):
        return Parameter(Tensor.rand(*args, **kwargs))

    @staticmethod
    def randn(*args, **kwargs):
        return Parameter(Tensor.randn(*args, **kwargs))

