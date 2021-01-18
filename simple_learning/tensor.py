from typing import Any, Optional

import numpy as np

from .grad.context import Context


class Tensor:
    """Array-like object to track operations and perform automatic gradient computation."""

    def __init__(self,
                 data: Any,
                 requires_grad: bool = True,
                 context: Optional[Context] = None):

        if isinstance(data, np.ndarray):
            self.data = data.astype('float32')
        elif isinstance(data, Tensor):
            self.data = data.data
        elif hasattr(data, '__iter__'):
            self.data = np.array(data, dtype='float32')
        else:
            self.data = np.array([data], dtype='float32')

        self.requires_grad = requires_grad
        self.context = context
        self.grad = None

    # simple utility
    def detach(self):
        return Tensor(self.data, requires_grad=False)

    # properties
    @property
    def shape(self):
        return self.data.shape

    @property
    def function(self):
        return getattr(self.context, 'function', None)

    @property
    def numel(self):
        return self.data.size

    # Tensor creation
    @staticmethod
    def ones(shape: tuple, requires_grad: bool = True):
        return Tensor(np.ones(shape), requires_grad=requires_grad)

    @staticmethod
    def zeros(shape: tuple, requires_grad: bool = True):
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def rand(*shape: int, requires_grad: bool = True):
        return Tensor(np.random.rand(*shape), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape: int, requires_grad: bool = True):
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

    # backprop
    def backward(self, grad=None):
        # fill the incoming gradient w.r.t. the Tensor with ones
        self.grad = grad
        if self.grad is None:
            self.grad = np.ones(self.data.shape)

        # gather all the nodes in the graph
        all_nodes = self._record_nodes(set(), [])

        # traverse each node, filling its corresponding parents' gradient
        for node in reversed(all_nodes):
            parents_grads = node.context.function.apply_backward(node.context, node.grad)
            for parent, p_grad in zip(node.context.parents, parents_grads):
                if p_grad is None:
                    continue

                assert p_grad.shape == parent.data.shape, f"Gradient shape {p_grad.shape} not the same"\
                                                          f"as input shape {parent.data.shape}"\
                                                          f"in function{node.context.function}"

                if parent.grad is None:
                    parent.grad = p_grad
                else:
                    parent.grad += p_grad

    def _record_nodes(self, visited: set, nodes: list):
        visited.add(self)

        assert self.context is not None, "Can't build graph on leaf Tensor as the root."

        for parent in self.context.parents:
            if parent not in visited and parent.context is not None:
                parent._record_nodes(visited, nodes)

        nodes.append(self)
        return nodes

    def __repr__(self):
        function = getattr(self.context, 'function', None)
        if function:
            function = function.__name__
        return f"<Tensor: {self.data} From function: {function}>"

    # basic operators
    # TODO: add the corresponding __r[op]__ and __i[op]__
    def __add__(self, other: Any):
        other = to_tensor(other)
        from .grad.functional import Add
        return Add.apply(self, other)

    def __sub__(self, other: Any):
        other = to_tensor(other)
        from .grad.functional import Sub
        return Sub.apply(self, other)

    def __rsub__(self, other):
        other = to_tensor(other)
        from .grad.functional import Sub
        return Sub.apply(other, self)

    def __mul__(self, other: Any):
        other = to_tensor(other)
        from .grad.functional import Mul
        return Mul.apply(self, other)

    def __matmul__(self, other: Any):
        other = to_tensor(other)
        from .grad.functional import Matmul
        return Matmul.apply(self, other)

    def __truediv__(self, other: Any):
        other = to_tensor(other)
        from .grad.functional import Div
        return Div.apply(self, other)

    def __pow__(self, other: Any):
        other = to_tensor(other)
        from .grad.functional import Pow
        return Pow.apply(self, other)

    def reshape(self, shape: tuple):
        shape = to_tensor(shape)
        from .grad.functional import Reshape
        return Reshape.apply(self, shape)

    def transpose(self, order=None):
        order = to_tensor(order)
        from .grad.functional import Transpose
        return Transpose.apply(self, order)

    # operations on self
    def mean(self):
        from .grad.functional import Mean
        return Mean.apply(self)

    def sum(self, axis=None, keepdims=True):
        from .grad.functional import SumSelf
        return SumSelf.apply(self, axis=axis, keepdims=keepdims)

    def log(self):
        from .grad.functional import Log
        return Log.apply(self)

    def exp(self):
        from .grad.functional import Exp
        return Exp.apply(self)

    # aliases
    def dot(self, other: Any):
        return self @ other

    def size(self):
        return self.shape

    @property
    def T(self):
        return self.transpose()

    def t(self):
        return self.transpose()


def to_tensor(obj: Any):
    """Convert any input to Tensor or return the object if it's already a Tensor.
    It makes the assumption that if the input wasn't already a tensor, tracking
    the gradient of that object is not desired."""
    if isinstance(obj, Tensor):
        return obj
    else:
        return Tensor(obj, requires_grad=False)

