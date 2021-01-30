from simple_learning.tensor import to_tensor

from .grad.functional import ReLU, SoftMax, CrossEntropy


# TODO: convert basic pseudo-code to LaTeX
def relu(tensor):
    """Apply the ReLU non-linearity elementwise to a Tensor.

       Pseudo-code: max(0, x)

    Args:
        tensor: Tensor of any size.
    """
    tensor = to_tensor(tensor)
    return ReLU.apply(tensor)


def softmax(tensor):
    """Apply the softmax function to the last dimension of a Tensor.

    Args:
        tensor: Tensor of any size.
    """
    tensor = to_tensor(tensor)
    return SoftMax.apply(tensor)


def cross_entropy(in_tensor, targets):
    """Apply the Cross Entropy function to the elements selected by "targets" and average
       the result.

       Pseudo-code: (-ln(in_tensor[i, targets[i]]) for i in range(in_tensor.shape[0])).mean()

    Args:
        in_tensor: Tensor of size (N, C), where N are the number of examples and
                   C the number of classes.
        targets: Tensor of size(C).

    Returns:
        Average of the loss across examples.
    """
    in_tensor = to_tensor(in_tensor)
    targets = to_tensor(targets)
    return CrossEntropy.apply(in_tensor, targets)

