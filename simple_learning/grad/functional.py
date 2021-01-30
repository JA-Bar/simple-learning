# basic functions are inspired by Tinygrad's own implementation
import numpy as np

from .function import Function


def unbroadcast(out, in_shape):
    """Sum the gradients of the output in the case that broadcasting was performed
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

    # finally, sum the axis where broadcasting took place
    sum_axis = tuple([dim for dim in range(len(in_shape)) if in_shape[dim]==1 and out.shape[dim]>1])
    return out.sum(axis=sum_axis).reshape(original_in_shape)


# basic tensor operations
class Add(Function):
    @staticmethod
    def forward(context, x1, x2):
        context.save_for_backward(x1.shape, x2.shape)
        return x1 + x2

    @staticmethod
    def backward(context, output_grads):
        # y = x1 + x2 ||| dy/dx1 = dy/dx2 = 1
        # the local gradient of the sum operator will be 1 for both inputs. Now just multiply the
        # local gradient with the incoming gradient to get the gradient of the target function
        # w.r.t. the inputs and keep the chain rule going
        x1_shape, x2_shape = context.saved_data
        return unbroadcast(output_grads, x1_shape), unbroadcast(output_grads, x2_shape)


class Sub(Function):
    @staticmethod
    def forward(context, x1, x2):
        context.save_for_backward(x1.shape, x2.shape)
        return x1 - x2

    @staticmethod
    def backward(context, output_grads):
        # y = x1 - x2 ||| dy/x1 = 1    dy/x2 = -1
        x1_shape, x2_shape = context.saved_data
        return unbroadcast(output_grads, x1_shape), unbroadcast(-output_grads, x2_shape)


class Mul(Function):
    @staticmethod
    def forward(context, x1, x2):
        context.save_for_backward(x1, x2)
        return x1 * x2

    @staticmethod
    def backward(context, output_grads):
        # y = x1 * x2 ||| dy/x1 = x2    dy/x2 = x1
        x1, x2 = context.saved_data
        return unbroadcast(x2 * output_grads, x1.shape), unbroadcast(x1 * output_grads, x2.shape)


class Div(Function):
    @staticmethod
    def forward(context, x1, x2):
        context.save_for_backward(x1, x2)
        return x1 / x2

    @staticmethod
    def backward(context, output_grads):
        # y = x1 / x2 ||| dy/x1 = (1x2)    dy/x2 = x1 * d(1/x2)/x2 = x1 * -(1/x2**2)
        x1, x2 = context.saved_data
        return (unbroadcast((1/x2) * output_grads, x1.shape),
                unbroadcast(x1 * (-1/x2**2) * output_grads, x2.shape))


class Pow(Function):
    @staticmethod
    def forward(context, x, y):
        context.save_for_backward(x, y)
        return x ** y

    @staticmethod
    def backward(context, output_grads):
        x, y = context.saved_data
        x_non_negative = x.copy()
        x_non_negative[x_non_negative < 0] = np.nan
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
        x_shape = x.shape
        y_shape = y.shape

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        if len(output_grads.shape) == 1:
            output_grads = np.expand_dims(output_grads, axis=0)

        x_grad = unbroadcast(output_grads@y.T, x.shape)
        y_grad = unbroadcast(x.T@output_grads, y.shape)

        return x_grad.reshape(x_shape), y_grad.reshape(y_shape)


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
        # the dimensions of the output grad are the ones from the original input
        # regardless of keepdims
        axis, input_shape, keepdims = context.saved_data
        if not keepdims and input_shape != (1,):
            output_grads = np.expand_dims(output_grads, axis)

        grads = np.zeros(input_shape, dtype='float32') + output_grads
        return grads.reshape(input_shape)


class Exp(Function):
    @staticmethod
    def forward(context, array):
        result = np.exp(array)
        context.save_for_backward(result)
        return result

    @staticmethod
    def backward(context, output_grads):
        forward_result, = context.saved_data
        return forward_result * output_grads


class Log(Function):
    @staticmethod
    def forward(context, array):
        context.save_for_backward(array)
        return np.log(array)

    @staticmethod
    def backward(context, output_grads):
        EPSILON = 1e-9
        forward_input, = context.saved_data
        return (1/(forward_input+EPSILON)) * output_grads


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


class SoftMax(Function):
    @staticmethod
    def forward(context, array):
        # if there are problems, look into the numerically stable implementation
        input_shape = array.shape
        n_dims = len(input_shape)

        # treat all vectors as column vectors
        if n_dims == 1:
            array = np.expand_dims(array, axis=0)
            n_dims = 2

        exp = np.exp(array)
        result = exp / np.sum(exp, axis=(n_dims-1), keepdims=True)

        context.save_for_backward(input_shape, result)

        return result.reshape(input_shape)

    @staticmethod
    def backward(context, output_grads):
        input_shape, forward_result = context.saved_data

        # great further explanation from https://stackoverflow.com/a/36280783
        # compute J[i, j] for i != j resulting in -softmax_i * softmax_j
        jacobian = -forward_result[..., np.newaxis] * forward_result[:, np.newaxis, :]

        # get the diagonal indices (i=j) and fill them with softmax_i * (1 - softmax_i)
        idx_y, idx_x = np.diag_indices_from(jacobian[0])
        jacobian[:, idx_y, idx_x] = forward_result * (1. - forward_result)

        # reduce the jacobian down to a gradient w.r.t. the inputs:
        # a column of the jacobian tells you how every output is affected by a particular input,
        # output_grads tell you how every output affects the target function,
        # so by multiplying output_grads by column j and summing the result
        # you will get the total influence of input j over all the outputs
        output_grads = output_grads[..., np.newaxis, :]
        return (output_grads @ jacobian).reshape(input_shape)


class CrossEntropy(Function):
    @staticmethod
    def forward(context, in_tensor, targets):
        # targets will be used as indices so integers are required
        targets = targets.astype('int')
        context.save_for_backward(in_tensor, targets)

        # select only the inputs that will affect the loss
        n = in_tensor.shape[0]
        inputs_in_target_indices = in_tensor[range(n), targets]

        # apply cross-entropy loss to those inputs and return the average
        log_result = -np.log(inputs_in_target_indices)
        return np.sum(log_result) * (1/n)

    @staticmethod
    def backward(context, output_grads):
        EPSILON = 1e-9
        in_tensor, targets = context.saved_data

        n = in_tensor.shape[0]

        # every local gradient will be 0, except the ones corresponding to the inputs
        # used to calculate the forward pass, those will have regular -1/x grad
        local_grads = np.zeros_like(in_tensor)
        local_grads[range(n), targets] = -1/(in_tensor[range(n), targets]+EPSILON)
        local_grads *= (1/n)
        return local_grads * output_grads


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


