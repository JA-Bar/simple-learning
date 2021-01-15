from simple_learning import Tensor

from .context import Context


class Function:
    """Interface for custom functions to be added to a computational graph.

    Any custom functions should inherit from Function and implement two mandatory
    static methods: forward and backward.

    The function should be called by using the apply method, which will return
    the resulting Tensor of the operation with its corresponding information to
    track the operations that created that Tensor if any of the Tensors that
    were used as arguments on the function have requires_grad=True.
    """
    @staticmethod
    def forward(context, *args, **kwargs):
        """Compute the forward pass of an operation.

        Args:
            context: Context object that contains the information to compute
                     the backward pass.
            *args: Input numpy arrays to the operation.
            **kwargs: Aditional options.

        Returns:
            The resulting numpy array of the operation.
        """
        raise NotImplementedError("The forward method should be implemented.")

    @staticmethod
    def backward(context, *output_grads):
        """Compute the backward pass of an operation.

        Args:
            context: Context object that contains the information to compute
                     the backward pass.
            output_grads: The numpy array corresponding to forward's output gradient.
        Returns:
            The gradient w.r.t. the input values of the function. The number of values
            returned should be the same as the number of inputs, in case the input doesn't
            require a gradient, return None in its place.
        """
        raise NotImplementedError("The backward method should be implemented.")

    @classmethod
    def apply(cls, *args, **kwargs):
        """Wrapper of the forward pass. Takes care of transforming the Tensor inputs to
        numpy arrays, generating the appropriate context, and returning another Tensor
        with the result of the operation.
        """
        assert all([isinstance(arg, Tensor) for arg in args]), "All positional args to a "\
                                                               "Function must be Tensors"

        requires_grad = any([tensor.requires_grad for tensor in args])

        context = Context(cls, parents=(args))

        args = [tensor.data for tensor in args]
        result = cls.forward(context, *args, **kwargs)

        if not requires_grad:
            context = None

        return Tensor(result, requires_grad=requires_grad, context=context)

    @classmethod
    def apply_backward(cls, context, *args, **kwargs):
        """Wrapper of the backward pass. Takes case of transforming the size of the output
        to tuple when only one gradient is returned from Function.backward, this way the
        implementation of backward in Tensor can work correctly.
        """
        result = cls.backward(context, *args, **kwargs)
        if not isinstance(result, (list, tuple)):
            result = (result, )
        return result

