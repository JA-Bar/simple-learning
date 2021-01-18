from simple_learning.tensor import to_tensor

from .grad.functional import ReLU, SoftMax


def relu(tensor):
    tensor = to_tensor(tensor)
    return ReLU.apply(tensor)


def softmax(tensor):
    tensor = to_tensor(tensor)
    return SoftMax.apply(tensor)

