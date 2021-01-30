from .optimizer import Optimizer


__all__ = [
    'SGD'
]


class SGD(Optimizer):
    """Implementation of the Stochastic Gradient Descent optimizer with an optional
    momentum parameter.

    Args:
        parameters: Iterable with the reference to all the parameters to be optimized. This
                    can be easily achieved by calling the get_parameters() method in any Module.
        learning_rate: Learning rate, defaults to 0.01.
        momentum: Optional momentum coefficient, if momentum = 0 it's equivalent to classical SGD.
    """
    def __init__(self, parameters, learning_rate=0.01, momentum=0.0, weight_decay=0):
        super().__init__(parameters)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        for param in self._parameters:
            assert param.grad is not None, "Can't update weights without calling backward first."
            param.buffer = self.momentum * param.buffer + param.grad  # momentum buffer
            param.data = param.data - self.learning_rate * param.buffer


