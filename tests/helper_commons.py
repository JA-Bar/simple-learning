import numpy as np
import numpy.testing as np_test

import torch

import simple_learning as sl

np.random.seed(42)

A_TOLERANCE = 1e-6
R_TOLERANCE = 1e-6


def evaluate_function_with_pytorch(simple_learning_function, torch_function, constructor):
    # initialize numpy arrays to be used as arguments of the function
    args_arrays = [constructor[0](*args) for args in constructor[1:]]

    # apply simple_learning function to Tensors
    sl_args = [sl.Tensor(arg.copy()) for arg in args_arrays]
    sl_result = simple_learning_function(*sl_args)
    sl_result.backward(np.ones_like(sl_result.data))

    # apply the same function to pytorch's Tensors
    pt_args = [torch.tensor(arg.copy().astype('float32'), requires_grad=True) for arg in args_arrays]
    pt_result = torch_function(*pt_args)
    pt_result.backward(torch.ones_like(pt_result))

    # check if the forward pass is correct
    obtained_result = sl_result.data
    expected_result = pt_result.detach().numpy()
    np_test.assert_allclose(obtained_result, expected_result, R_TOLERANCE, A_TOLERANCE)

    # check if the backward pass if correct (the arguments' gradients)
    for sl_a, pt_a in zip(sl_args, pt_args):
        obtained_grad = sl_a.grad
        expected_grad = pt_a.grad.numpy()
        np_test.assert_allclose(obtained_grad, expected_grad, R_TOLERANCE, A_TOLERANCE)


