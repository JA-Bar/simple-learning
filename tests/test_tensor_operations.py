import operator

import numpy as np
import numpy.testing as np_test

import pytest
import torch

import simple_learning as sl

from helper_commons import evaluate_function_with_pytorch

np.random.seed(42)

# absolute and relative tolerances
A_TOLERANCE = 1e-6
R_TOLERANCE = 1e-6

pytestmark = pytest.mark.core


# operations between two tensors
class TestOperationsTwoTensors:
    scenario_1 = ('scalar', {'constructor': [np.random.randn, (1, ), (1, )]})
    scenario_2 = ('one_dimensional', {'constructor': [np.random.randn, (5, ), (5, )]})
    scenario_3 = ('row_vector', {'constructor': [np.random.randn, (1, 5), (1, 5)]})
    scenario_4 = ('multiple_vectors', {'constructor': [np.random.randn, (10, 5), (10, 5)]})

    scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

    def test_sum(self, constructor):
        evaluate_function_with_pytorch(operator.add, operator.add, constructor)

    def test_sub(self, constructor):
        evaluate_function_with_pytorch(operator.sub, operator.sub, constructor)

    def test_mul(self, constructor):
        evaluate_function_with_pytorch(operator.mul, operator.mul, constructor)

    def test_div(self, constructor):
        evaluate_function_with_pytorch(operator.truediv, operator.truediv, constructor)

    def test_pow(self, constructor):
        constructor[0] = np.random.rand  # overriding function to keep the tests real valued
        evaluate_function_with_pytorch(operator.pow, operator.pow, constructor)

    def test_matmul(self, constructor):
        constructor[-1] = constructor[-1][::-1]  # change the size of last argument to match matmul
        evaluate_function_with_pytorch(operator.matmul, operator.matmul, constructor)


# # operations on the Tensor itself
class TestOperationsOneTensor:
    scenario_1 = ('scalar', {'constructor': [np.random.randn, (1, )]})
    scenario_2 = ('one_dimensional', {'constructor': [np.random.randn, (5, )]})
    scenario_3 = ('row_vector', {'constructor': [np.random.randn, (1, 5)]})
    scenario_4 = ('multiple_vectors', {'constructor': [np.random.randn, (10, 5)]})

    scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

    def test_mean(self, constructor):
        evaluate_function_with_pytorch(lambda x: x.mean(), lambda x: x.mean(), constructor)

    def test_reshape(self, constructor):
        evaluate_function_with_pytorch(lambda x: x.reshape((-1, 1)),
                                       lambda x: x.reshape((-1, 1)), constructor)

    def test_transpose(self, constructor):
        evaluate_function_with_pytorch(lambda x: x.T, lambda x: x.T, constructor)

    def test_sum_self(self, constructor):
        evaluate_function_with_pytorch(lambda x: x.sum(axis=0, keepdims=True),
                                       lambda x: x.sum(axis=0, keepdims=True), constructor)

    def test_exp(self, constructor):
        evaluate_function_with_pytorch(lambda x: x.exp(), lambda x: x.exp(), constructor)

    def test_log(self, constructor):
        constructor[0] = np.random.rand  # overriding function to keep the tests real valued
        evaluate_function_with_pytorch(lambda x: x.log(), lambda x: x.log(), constructor)


# # helper function to compose operations and check the result against pytorch
def compose_operations(*functions_and_args):
    sl_result = None
    pt_result = None
    sl_leaves = []
    pt_leaves = []

    for i, (function, args) in enumerate(functions_and_args):
        if not isinstance(args, (tuple, list)):
            args = (args,)

        # convert arguments to Tensors
        sl_args = [sl.Tensor(arg.copy()) for arg in args]
        pt_args = [torch.tensor(arg.copy().astype('float32'), requires_grad=True) for arg in args]

        # add the arguments to the list of leaf Tensors
        sl_leaves.extend(sl_args)
        pt_leaves.extend(pt_args)

        # if it's the first operation use the provided arguments only
        if i == 0:
            sl_result = function(*sl_args)
            pt_result = function(*pt_args)
        # otherwise, use the output of the last operation as the first argument
        else:
            print(sl_args)
            sl_result = function(sl_result, *sl_args)
            pt_result = function(pt_result, *pt_args)

    # compute the backward pass
    sl_result.backward(np.ones_like(sl_result.data))
    pt_result.backward(torch.ones_like(pt_result))

    # check if the forward pass is correct
    obtained_result = sl_result.data
    expected_result = pt_result.detach().numpy()
    np_test.assert_allclose(obtained_result, expected_result, R_TOLERANCE, A_TOLERANCE)

    # check if the backward pass if correct (the arguments' gradients)
    for sl_a, pt_a in zip(sl_leaves, pt_leaves):
        obtained_grad = sl_a.grad
        expected_grad = pt_a.grad.numpy()
        np_test.assert_allclose(obtained_grad, expected_grad, R_TOLERANCE, A_TOLERANCE)


def test_composition_1():
    compose_operations(
        [operator.add, (np.random.rand(3, 2), np.random.rand(3, 2))],
        [operator.mul, (np.random.rand(3, 2))],
        [operator.pow, (np.random.randn(3, 2))],
        [operator.matmul, (np.random.randn(2, 3))],
        [lambda x: x.reshape((-1, 1)), ()],
        [lambda x: x.mean(), ()],
    )

