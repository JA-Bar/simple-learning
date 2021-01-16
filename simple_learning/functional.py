from .grad.functional import ReLU


def relu(tensor):
    return ReLU.apply(tensor)

