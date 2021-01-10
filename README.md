# Simple Learning

A basic, bare-bones, CPU-bound, DL framework with the objective of learning how things work.

Inspired by [Piotr Skalski's DL projects](https://github.com/SkalskiP/ILearnDeepLearning.py),
[George Hotz's Tinygrad](https://github.com/geohot/tinygrad),
[Andrej Karpathy's Micrograd](https://github.com/karpathy/micrograd),
and of course, [pytorch](https://github.com/pytorch/pytorch).


## The purpose

This library was created to closely resemble the pytorch interface and structure, but
with a simpler implementation and restricted to CPU use. Its implementation 100% in
python and explicit code will hopefully provide a clearer view of how a library of this
type may be implemented without getting lost in code optimization, additional options,
or confusing program flow.

Many of the functions and ideas also take direct inspiration from Tinygrad, another
simple framework with more functionality and capability than this library.

This is an ongoing experimentation project and many features are still in the works.


## Basics

### Tensors

The whole library is built around Tensors, wrappers of numpy arrays that allow every
operation to be tracked, so automatic gradient calculation can be performed. Tensors
try to mimic pytorch, so the basic interface is very similar.

```python3
from simple_learning import Tensor


# create Tensors
tensor_1 = Tensor([1, 2, 3, 4])
tensor_2 = Tensor([5, 6, 7, 8])

# operations including Tensors will create other Tensors
tensor_3 = tensor_1 * 10

# call backward to calculate gradients
result = tensor_2 - tensor_3
result = result.mean()
result.backward()

# Voila
print(tensor_1.grad)
print(tensor_2.grad)
```

For the sake of simplicity every Tensor is of type float32 and gradient is tracked
by default (controlled by requires_grad).


### Functions

As stated before, every function that has at least one Tensor creates another Tensor.
These functions are all created under a simple interface very similar to pytorch that
allows operation tracking, so a user can implement a custom function by inheriting
from a Function class and implementing a forward and backward methods that describe
how the result of the function and the gradient are calculated respectively.

```python3
from simple_learning.function import Function


class CustomFunction(Function):
    @staticmethod
    def forward(context, my_np_array):
        context.save_for_backward(my_np_array.shape)
        return my_np_array + 42

    @staticmethod
    def backward(context, output_grads):
        saved_shape, = context.saved_data
        return output_grads.reshape(saved_shape)
```

The function can now be called by using `CustomFunction.apply(any_Tensor)` to calculate
the forward pass and simple_learning will automatically call backward when needed.

## Tests

The current tests cover the basic Tensor operations and can be called by using
`pytest`.


