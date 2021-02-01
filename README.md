# Simple Learning

A simple, bare-bones, CPU-bound, DL framework with the objective of learning how things work.

Inspired by [Piotr Skalski's DL projects](https://github.com/SkalskiP/ILearnDeepLearning.py),
[George Hotz's Tinygrad](https://github.com/geohot/tinygrad),
[Andrej Karpathy's Micrograd](https://github.com/karpathy/micrograd),
and of course, [PyTorch](https://github.com/pytorch/pytorch).


## The purpose

This library was created to closely resemble the PyTorch interface and structure, but
with a simpler implementation and restricted to CPU use. Its implementation 100% in
python and explicit code will hopefully provide a clearer view of how a library of this
type may be implemented without getting lost in code optimization, additional options,
or confusing program flow.

Many sections of the code are commented with the ideas to understand what's happening.
The general overview of the project and the interaction between objects will be outlined in
a document in the future.

Many of the functions and ideas also take direct inspiration from Tinygrad, another
simple framework, but with a focus on GPU as well as Apple's Neural Engine.

This is an ongoing experimentation project and many features are still in the works.


## Basics

### Tensors

The whole library is built around Tensors, wrappers of numpy arrays that allow every
operation to be tracked, so automatic gradient calculation can be performed. Tensors
try to mimic pytorch, so the basic interface is very similar.

```python3
from simple_learning.tensor import Tensor


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
allows operation tracking, so a user can implement a custom function by subclassing
a Function and implementing a forward and backward methods that describe
how the result of the function and the gradient are calculated respectively.

```python3
from simple_learning.function import Function


class CustomFunction(Function):
    # define how to compute the output
    @staticmethod
    def forward(context, my_np_array):
        # you can save objects for the backward pass using the context
        context.save_for_backward(my_np_array.shape)
        return my_np_array + 42

    # define how to compute the gradient
    @staticmethod
    def backward(context, output_grads):
        saved_shape, = context.saved_data
        return output_grads.reshape(saved_shape)
```

The function can now be called by using `CustomFunction.apply(any_Tensor)` to calculate
the forward pass and simple_learning will automatically call backward when needed.


### Modules

Just like in PyTorch, Modules are the base for building more complex computing units that
depend on a certain state to calculate the output (i.e. parameters), or simply as a way to
further organize operations or other Modules.

To create a custom module, you first have to subclass Module, call its init, and define
a forward method.

```python3
import simple_learning.nn as nn
import simple_learning.functional as F


class CustomModule(nn.Module):
    # initialize the module
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.fc_1 = nn.Linear(input_size, 10)
        self.fc_2 = nn.Linear(10, output_size)

    # describe your computation of the output
    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        
        x = self.fc_2(x)
        x = F.relu(x)

        return F.softmax(x)


model = CustomModule(20, 2)
result = model(some_input_tensor)
```

The output can then be calculated by using the module as a callable on any input Tensor.


### Optimizers

An optimizer is used by initializing it with the parameters to optimize, this can be
achieved by calling the .get_parameters() method on any Module, then the usual
.zero_grad() and .step() methods can be used to update parameters.

```python3
import simple_learning.functional as F
from simple_learning.optim import SGD


optimizer = SGD(model.get_parameters(), learning_rate, momentum)

logits = model(some_input_tensor)
loss = F.cross_entropy(logits, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Extra utilities

There are a couple of extra features such as a logger to easily keep track of training
and the no_grad context manager to not compute gradients. These set of features won't
necessarily mimic PyTorch's own, but will add convenience when training models or viewing
data.


## Requirements

Numpy, ... that's it.

In case you want to run the tests pytest and PyTorch will also need to be installed, but they
are not a requirement to run the library.


## Tests

The current tests cover the basic Tensor operations, interfaces, functions, and some Modules.
They  can be called by using `pytest`.

