import numpy as np
import numpy.testing as np_test

import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_learning import Tensor
from simple_learning.nn.modular import Linear
from simple_learning.functional import relu


np.random.seed(42)
# absolute and relative tolerances
A_TOLERANCE = 1e-6
R_TOLERANCE = 1e-6


def test_linear():
    in_features = 10
    out_features = 20
    batch = 30
    input_array = np.random.randn(batch, in_features).astype('float32')

    pt_linear = nn.Linear(in_features, out_features, bias=True)
    pt_input = torch.tensor(input_array.copy(), requires_grad=True)
    pt_result = pt_linear(pt_input)
    pt_result.backward(torch.ones_like(pt_result))

    sl_linear = Linear(in_features, out_features, bias=True)
    sl_linear.weight.data = pt_linear.weight.data.numpy()
    sl_linear.bias.data = pt_linear.bias.data.numpy()
    sl_input = Tensor(input_array.copy(), requires_grad=True)
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


def test_relu():
    input_array = np.random.randn(20, 30).astype('float32')

    pt_input = torch.tensor(input_array.copy(), requires_grad=True)
    pt_result = F.relu(pt_input)
    pt_result.backward(torch.ones_like(pt_result))

    sl_input = Tensor(input_array.copy(), requires_grad=True)
    sl_result = relu(sl_input)
    sl_result.backward()

    obtained_result = sl_result.data
    expected_result = pt_result.detach().numpy()
    np_test.assert_allclose(obtained_result, expected_result, R_TOLERANCE, A_TOLERANCE)

    obtained_grad = sl_input.grad
    expected_grad = pt_input.grad.numpy()
    np_test.assert_allclose(obtained_grad, expected_grad, R_TOLERANCE, A_TOLERANCE)


