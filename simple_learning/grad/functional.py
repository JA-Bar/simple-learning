# most basic functions are clones of Tinygrad's own implementation
import numpy as np

from .function import Function


def unbroadcast(out, in_shape):
    """Sums the gradients of the output in the case that broadcasting was performed
    during the calculation of a result. This effectively avoids explicitly splitting
    a broadcasting operation into several clone modules beforehand.
    """
    # if the input is a scalar, sum over every dimension
    if in_shape == (1,):
        sum_axis = None
        return out.sum(axis=sum_axis).reshape(in_shape)

    original_in_shape = in_shape

    # if it's an (n,) shape vector change its shape to mimic (1, n, 1, ...) according to output shape
    if len(in_shape) == 1:
        n = in_shape[0]
        index = out.shape[::-1].index(n)
        temp_axis = [n if i == index else 1 for i in range(len(out.shape))]
        in_shape = temp_axis[::-1]

    sum_axis = tuple([dim for dim in range(len(in_shape)) if in_shape[dim]==1 and out.shape[dim]>1])
    return out.sum(axis=sum_axis).reshape(original_in_shape)


# basic tensor operations
class Add(Function):
    @staticmethod
    def forward(context, x, y):
        context.save_for_backward(x.shape, y.shape)
        return x + y

    @staticmethod
    def backward(context, output_grads):
        x_shape, y_shape = context.saved_data
        return unbroadcast(output_grads, x_shape), unbroadcast(output_grads, y_shape)


class Sub(Function):
    @staticmethod
    def forward(context, x, y):
        context.save_for_backward(x.shape, y.shape)
        return x - y

    @staticmethod
    def backward(context, output_grads):
        x_shape, y_shape = context.saved_data
        return unbroadcast(output_grads, x_shape), unbroadcast(-output_grads, y_shape)


class Mul(Function):
    @staticmethod
    def forward(context, x, y):
        context.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(context, output_grads):
        x, y = context.saved_data
        return unbroadcast(y * output_grads, x.shape), unbroadcast(x * output_grads, y.shape)


class Div(Function):
    @staticmethod
    def forward(context, x, y):
        context.save_for_backward(x, y)
        return x / y

    @staticmethod
    def backward(context, output_grads):
        x, y = context.saved_data
        return (unbroadcast((1/y) * output_grads, x.shape),
                unbroadcast(x * (-1/y**2) * output_grads, y.shape))


class Pow(Function):
    @staticmethod
    def forward(context, x, y):
        context.save_for_backward(x, y)
        return x ** y

    @staticmethod
    def backward(context, output_grads):
        x, y = context.saved_data
        x_non_negative = x.copy()
        x_non_negative[x_non_negative<0] = np.nan
        return (unbroadcast(y * (x**(y-1.0)) * output_grads, x.shape),
                unbroadcast((x**y) * np.log(x_non_negative) * output_grads, y.shape))


class Matmul(Function):
    @staticmethod
    def forward(context, x, y):
        context.save_for_backward(x, y)
        return x @ y

    @staticmethod
    def backward(context, output_grads):
        x, y = context.saved_data
        return unbroadcast(output_grads@y.T, x.shape), unbroadcast(x.T@output_grads, y.shape)


class Mean(Function):
    @staticmethod
    def forward(context, array):
        div_coeff = 1 / array.size
        context.save_for_backward(div_coeff, array.shape)

        pre_sum = array * div_coeff
        return pre_sum.sum()

    @staticmethod
    def backward(context, output_grads):
        div_coeff, input_shape = context.saved_data
        weighted_grads = output_grads * div_coeff
        return np.ones(input_shape) * weighted_grads


class Reshape(Function):
    @staticmethod
    def forward(context, array, shape):
        shape = shape.astype('int')
        context.save_for_backward(array.shape)
        return array.reshape(shape)

    @staticmethod
    def backward(context, output_grads):
        input_shape, = context.saved_data
        return output_grads.reshape(input_shape)


class Transpose(Function):
    @staticmethod
    def forward(context, array, order):
        if np.isnan(order).all():
            order = None
        else:
            order = order.astype('int')

        context.save_for_backward(order)
        return array.transpose(order)

    @staticmethod
    def backward(context, output_grads):
        order, = context.saved_data
        if order is None:
            return output_grads.transpose()
        un_transpose = [order[idx] for idx in order]
        return output_grads.transpose(un_transpose)


class SumSelf(Function):
    @staticmethod
    def forward(context, array, axis=None, keepdims=False):
        context.save_for_backward(axis, array.shape, keepdims)
        return array.sum(axis, keepdims=keepdims, dtype='float32')

    @staticmethod
    def backward(context, output_grads):
        axis, input_shape, keepdims = context.saved_data
        if not keepdims and input_shape != (1,):
            output_grads = np.expand_dims(output_grads, axis)
        return np.zeros(input_shape, dtype='float32') + output_grads


# nn functions
class ReLU(Function):
    @staticmethod
    def forward(context, array):
        mask = array > 0
        context.save_for_backward(mask)
        return array * mask

    @staticmethod
    def backward(context, output_grads):
        mask, = context.saved_data
        return output_grads * mask


# nn module operations
class Linear(Function):
    # i = out_features
    # j = in_features
    # m = number of examples in the batch
    @staticmethod
    def forward(context, array, weight, bias):
        context.save_for_backward(array, weight, bias.shape)
        return array @ weight.T + bias  # Y[mxi] = X[mxj] @ W.T[jxi] + b[1xi]

    @staticmethod
    def backward(context, output_grads):
        array, weight, bias_shape = context.saved_data
        dX = output_grads @ weight  # dJ/dX[mxj] = dJ/dY[mxi] @ W[ixj]
        dW = output_grads.T @ array  # dJ/dW[ixj] = dJ/dY.T[ixm] @ X[mxj]
        db = unbroadcast(output_grads, bias_shape)  # dJ/db[ix1] = unbroadcast(dJ/db, b.shape)
        return dX, dW, db


class NaiveConv2d(Function):
    @staticmethod
    def forward(context, array, weight, stride, padding):
        pass

    @staticmethod
    def backward(context, output_grads):
        pass


class Conv2d(Function):
    @staticmethod
    def forward(context, array, weight, stride, padding):
        pass

    @staticmethod
    def backward(context, output_grads):
        pass


