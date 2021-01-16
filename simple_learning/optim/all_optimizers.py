from .optimizer import Optimizer


__all__ = [
    'SGD'
]


class SGD(Optimizer):
    def __init__(self, parameters, learning_rate=0.01, momentum=0.9, weight_decay=0):
        super().__init__(parameters)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    # TODO: weight decay
    # TODO: check averaging gradients
    def step(self):
        for param in self._parameters:
            assert param.grad is not None, "Can't update weights without calling backward first."
            param.buffer = self.momentum * param.buffer + param.grad
            param.data = param.data - self.learning_rate * param.buffer


