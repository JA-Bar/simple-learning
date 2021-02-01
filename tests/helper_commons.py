from dataclasses import dataclass
from itertools import repeat
from typing import Union, List, Tuple

import numpy as np
import numpy.testing as np_test

import torch

import simple_learning as sl

np.random.seed(42)

A_TOLERANCE = 1e-6
R_TOLERANCE = 1e-6


@dataclass
class Constructor:
    functions: Union[List, Tuple]
    arguments: Union[List, Tuple]


def evaluate_function_with_pytorch(simple_learning_function, torch_function, constructor):
    """Apply a simple_learning_function and a torch_function to the same numpy array based Tensors
       built with constructor, compare the results of both and the gradients of the leaf tensors.

    Args:
        simple_learning_function: Callable to evaluate the from the simple_learning library.
        torch_function: Callable to evaluate from the pytorch library.
        constructor: Pair of iterables ((func1, func2), (args_to_func1, args_to_func2)) to build
                     the numpy arrays used as parameters to both functions.

                     The arrays are initiated as func1(*args_to_func1), func2(*args_to_func2). The
                     result to each func will be a parameter to be used in both simple_learning and pytorch
                     functions.

                     In the case that only one constructor function is provided, but multiple
                     argument iterables, the function will be broadcasted to all other argument iterables:

                     Given: ((func1,), (args_to_func1, more_args_to_func1))
                     It's equivalent to: func1(*args_to_func1), func1(*more_args_to_func1).

    Raises:
        AssertionError if the simple_learning and pytorch functions results differ by more than the
        set absolute or relative tolerances.
    """
    constructor = Constructor(*constructor)
    if not isinstance(constructor.functions, (list, tuple)):
        constructor.functions = (constructor.functions, )

    # if the number of functions doesn't match the number of given argument iterables, repeat
    # broadcast the first given function to all args
    n_functions = len(constructor.functions)
    n_arguments = len(constructor.arguments)
    if n_functions < n_arguments:
        constructor.functions = repeat(constructor.functions[0], n_arguments)

    args_arrays = [func(*args) for (func, args) in zip(constructor.functions, constructor.arguments)]

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
        if sl_a.grad is None and pt_a.grad is None:
            continue  # if neither of the Tensors required grad, skip them

        obtained_grad = sl_a.grad
        expected_grad = pt_a.grad.numpy()
        np_test.assert_allclose(obtained_grad, expected_grad, R_TOLERANCE, A_TOLERANCE)


