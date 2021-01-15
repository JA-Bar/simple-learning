import inspect
import os
import sys

import numpy as np
import numpy.testing as np_test
import pytest
import torch

sys.path.insert(0, os.path.abspath('.'))
from simple_learning.tensor import Tensor

np.random.seed(42)
# absolute and relative tolerances
A_TOLERANCE = 1e-6
R_TOLERANCE = 1e-6

pytestmark = pytest.mark.core

# port of Robert Collins' test scenarios to check different cases easily (only on classes)
def pytest_generate_tests(metafunc):
    if metafunc.cls:
        idlist = []
        argvalues = []
        for scenario in metafunc.cls.scenarios:
            idlist.append(scenario[0])
            items = scenario[1].items()
            argnames = [x[0] for x in items]
            argvalues.append([x[1] for x in items])
        metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


# helper function to compare simple_learning and pytorch results
def evaluate_simple_operations(constructor_function, constructor_args, function, verbose=False):
    # initialize numpy arrays to be used as arguments of the function
    args_arrays = [constructor_function(*args) for args in constructor_args]

    # apply simple_learning function to Tensors
    sl_args = [Tensor(arg.copy()) for arg in args_arrays]
    sl_result = function(*sl_args)
    sl_result.backward(np.ones_like(sl_result.data))

    # apply the same function to pytorch's Tensors
    pt_args = [torch.tensor(arg.copy().astype('float32'), requires_grad=True) for arg in args_arrays]
    pt_result = function(*pt_args)
    pt_result.backward(torch.ones_like(pt_result))

    # check if the forward pass is correct
    obtained_result = sl_result.data
    expected_result = pt_result.detach().numpy()
    np_test.assert_allclose(obtained_result, expected_result, R_TOLERANCE, A_TOLERANCE)
    if verbose:
        print('Actual result: ', obtained_result)
        print('Expected result: ', expected_result)

    # check if the backward pass if correct (the arguments' gradients)
    for sl_a, pt_a in zip(sl_args, pt_args):
        obtained_grad = sl_a.grad
        expected_grad = pt_a.grad.numpy()
        np_test.assert_allclose(obtained_grad, expected_grad, R_TOLERANCE, A_TOLERANCE)
        if verbose:
            print('Actual gradient: ', obtained_grad)
            print('Expected gradient: ', expected_grad)


# operations between two tensors
class TestOperationsTwoTensors:
    scenario_1 = ('same_shape', {'constructor_function': np.random.randn,
                                 'constructor_args': [(3, 2), (3, 2)]})
    scenario_2 = ('broadcasting', {'constructor_function': np.random.randn,
                                   'constructor_args': [(10, 5, 1), (10, 5, 3)]})
    scenario_3 = ('with_scalar', {'constructor_function': np.random.randn,
                                  'constructor_args': [(10, 5, 3), (1,)]})

    scenarios = [scenario_1, scenario_2, scenario_3]

    def test_sum(self, constructor_function, constructor_args):
        evaluate_simple_operations(constructor_function, constructor_args, lambda x, y: x + y)

    def test_sub(self, constructor_function, constructor_args):
        evaluate_simple_operations(constructor_function, constructor_args, lambda x, y: x - y)

    def test_mul(self, constructor_function, constructor_args):
        evaluate_simple_operations(constructor_function, constructor_args, lambda x, y: x * y)

    def test_div(self, constructor_function, constructor_args):
        evaluate_simple_operations(constructor_function, constructor_args, lambda x, y: x / y)

    def test_pow(self, constructor_function, constructor_args):
        constructor_function = np.random.rand  # overriding function to keep the tests real valued
        evaluate_simple_operations(constructor_function, constructor_args, lambda x, y: x**y)


# specific dimensions for matmul
@pytest.mark.parametrize('constructor_function, constructor_args', [
                         (np.random.randn, [(3, 2), (2, 3)]),
                         (np.random.randn, [(3, 1), (1, 3)])
])
def test_matmul(constructor_function, constructor_args):
    evaluate_simple_operations(constructor_function, constructor_args, lambda x, y: x @ y)


# operations on the Tensor itself
class TestOperationsOneTensor:
    scenario_1 = ('scalar', {'constructor_function': np.random.randn,
                             'constructor_args': [(1,)]})
    scenario_2 = ('two_dims', {'constructor_function': np.random.randn,
                               'constructor_args': [(2, 3)]})
    scenario_3 = ('three_dims', {'constructor_function': np.random.randn,
                                 'constructor_args': [(10, 5, 3)]})

    scenarios = [scenario_1, scenario_2, scenario_3]

    def test_mean(self, constructor_function, constructor_args):
        evaluate_simple_operations(constructor_function, constructor_args, lambda x: x.mean())

    def test_reshape(self, constructor_function, constructor_args):
        evaluate_simple_operations(constructor_function, constructor_args, lambda x: x.reshape((1, -1)))

    def test_transpose(self, constructor_function, constructor_args):
        evaluate_simple_operations(constructor_function, constructor_args, lambda x: x.T)

    def test_sum_self(self, constructor_function, constructor_args):
        evaluate_simple_operations(constructor_function, constructor_args, lambda x: x.sum(axis=0, keepdims=True))


# helper function to compose operations and check the result against pytorch
def compose_operations(*functions_and_args):
    sl_result = None
    pt_result = None
    sl_leaves = []
    pt_leaves = []

    for i, (function, args) in enumerate(functions_and_args):
        if not isinstance(args, (tuple, list)):
            args = (args,)

        # convert arguments to Tensors
        sl_args = [Tensor(arg.copy()) for arg in args]
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


all_operations = {
    'sum': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'pow': lambda x, y: x**y,
    'matmul': lambda x, y: x @ y,
    'mean': lambda x: x.mean(),
    'reshape': lambda x: x.reshape((1, -1)),
    'transpose': lambda x: x.T,
    'sum_self': lambda x: x.sum(axis=0, keepdims=True)
}


def test_composition_1():
    compose_operations(
        [all_operations['sum'], (np.random.rand(3, 2), np.random.rand(3, 2))],
        [all_operations['mul'], (np.random.rand(3, 2))],
        [all_operations['pow'], (np.random.randn(3, 2))],
        [all_operations['matmul'], (np.random.randn(2, 3))],
        [all_operations['reshape'], ()],
        [all_operations['mean'], ()],
    )


# debugging
if __name__ == '__main__':
    module_functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    testing_functions = [name for name, value in module_functions if name.startswith('test_')]
    print(testing_functions)

