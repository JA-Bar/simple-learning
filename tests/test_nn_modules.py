import numpy as np
import numpy.testing as np_test

import torch
import torch.nn as torch_nn

import simple_learning as sl
import simple_learning.nn as sl_nn


np.random.seed(42)

# absolute and relative tolerances
A_TOLERANCE = 1e-6
R_TOLERANCE = 1e-6


def test_linear():
    in_features = 10
    out_features = 20
    batch = 30
    input_array = np.random.randn(batch, in_features).astype('float32')

    pt_linear = torch_nn.Linear(in_features, out_features, bias=True)
    pt_input = torch.tensor(input_array.copy(), requires_grad=True)
    pt_result = pt_linear(pt_input)
    pt_result.backward(torch.ones_like(pt_result))

    sl_linear = sl_nn.Linear(in_features, out_features, bias=True)
    sl_linear.weight.data = pt_linear.weight.data.numpy()
    sl_linear.bias.data = pt_linear.bias.data.numpy()
    sl_input = sl.Tensor(input_array.copy(), requires_grad=True)
    sl_result = sl_linear(sl_input)
    sl_result.backward()

    # check if forward pass is correct
    obtained_result = sl_result.data
    expected_result = pt_result.detach().numpy()
    np_test.assert_allclose(obtained_result, expected_result, R_TOLERANCE, A_TOLERANCE)

    sl_leaves = [sl_input, sl_linear.weight, sl_linear.bias]
    pt_leaves = [pt_input, pt_linear.weight, pt_linear.bias]

    # check if the backward pass if correct (the arguments' gradients)
    for sl_a, pt_a in zip(sl_leaves, pt_leaves):
        obtained_grad = sl_a.grad
        expected_grad = pt_a.grad.numpy()
        np_test.assert_allclose(obtained_grad, expected_grad, R_TOLERANCE, A_TOLERANCE)


def test_linear_2():
    in_features = 10
    out_features = 20
    batch = 30
    input_array = np.random.randn(batch, in_features).astype('float32')

    pt_linear = torch_nn.Linear(in_features, out_features, bias=False)
    pt_input = torch.tensor(input_array.copy(), requires_grad=True)
    pt_result = pt_linear(pt_input)
    pt_result.backward(torch.ones_like(pt_result))

    sl_linear = sl_nn.Linear(in_features, out_features, bias=False)
    sl_linear.weight.data = pt_linear.weight.data.numpy()
    sl_input = sl.Tensor(input_array.copy(), requires_grad=True)
    sl_result = sl_linear(sl_input)
    sl_result.backward()

    # check if forward pass is correct
    obtained_result = sl_result.data
    expected_result = pt_result.detach().numpy()
    np_test.assert_allclose(obtained_result, expected_result, R_TOLERANCE, A_TOLERANCE)

    sl_leaves = [sl_input, sl_linear.weight]
    pt_leaves = [pt_input, pt_linear.weight]

    # check if the backward pass if correct (the arguments' gradients)
    for sl_a, pt_a in zip(sl_leaves, pt_leaves):
        obtained_grad = sl_a.grad
        expected_grad = pt_a.grad.numpy()
        np_test.assert_allclose(obtained_grad, expected_grad, R_TOLERANCE, A_TOLERANCE)


def test_cross_entropy():
    batch = 20
    categories = 10

    from torch.nn.functional import cross_entropy as pt_ce
    from simple_learning.functional import cross_entropy as sl_ce
    from simple_learning.functional import softmax as sl_sm

    input_array = np.random.rand(batch, categories).astype('float32')
    targets = np.random.randint(0, categories, size=batch)

    pt_input = torch.tensor(input_array.copy(), requires_grad=True)
    pt_targets = torch.tensor(targets.copy()).long()

    pt_result = pt_ce(pt_input, pt_targets)
    pt_result.backward(torch.ones_like(pt_result))

    sl_input = sl.Tensor(input_array.copy(), requires_grad=True)
    sl_targets = sl.Tensor(targets.copy())

    sl_result = sl_sm(sl_input)
    sl_result = sl_ce(sl_result, sl_targets)
    sl_result.backward()

    # check if forward pass is correct
    obtained_result = sl_result.data
    expected_result = pt_result.detach().numpy()
    np_test.assert_allclose(obtained_result, expected_result, R_TOLERANCE, A_TOLERANCE)

    sl_leaves = [sl_input]
    pt_leaves = [pt_input]

    # check if the backward pass if correct (the arguments' gradients)
    for sl_a, pt_a in zip(sl_leaves, pt_leaves):
        obtained_grad = sl_a.grad
        expected_grad = pt_a.grad.numpy()
        np_test.assert_allclose(obtained_grad, expected_grad, R_TOLERANCE, A_TOLERANCE)


