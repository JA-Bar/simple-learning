import pytest

from simple_learning import Tensor
from simple_learning.grad.context import Context
from simple_learning.grad.function import Function, no_grad
from simple_learning.nn.module import Module
from simple_learning.nn.parameter import Parameter


def test_function_forward():
    with pytest.raises(NotImplementedError):
        Function.forward(Context(None))


def test_function_backward():
    with pytest.raises(NotImplementedError):
        Function.backward(Context(None))


def test_module_forward():
    with pytest.raises(NotImplementedError):
        Module.forward(None, Context(None))


def test_module_setgetattr():
    module = Module()

    param = Parameter([1, 2, 3])
    module.param = param
    assert module._parameters.get('param') is param

    child_module = Module()
    module.child_module = child_module
    assert module._modules.get('child_module') is child_module

    any_attr = [1, 2, 3]
    module.any_attr = any_attr
    assert module.__dict__.get('any_attr') is any_attr

    assert module.param is param
    assert module.child_module is child_module
    assert module.any_attr is any_attr

    with pytest.raises(AttributeError):
        module.some_missing_attr


@pytest.fixture
def parent_module():
    class ChildModule(Module):
        def __init__(self):
            super().__init__()

    class ParentModule(Module):
        def __init__(self):
            super().__init__()
            self.child_module = ChildModule()

    return ParentModule()


def test_module_get_parameters(parent_module):
    param_in_child = Parameter([1, 2, 3])
    param_in_parent = Parameter([1, 2, 3])

    parent_module.param = param_in_parent
    parent_module.child_module.param = param_in_child

    named_params = parent_module.get_named_parameters()
    parent_name = f'{parent_module.__module__}.ParentModule'
    assert named_params.get(f'{parent_name}.param') is param_in_parent
    assert named_params.get(f'{parent_name}.child_module.param') is param_in_child

    params = parent_module.get_parameters()
    assert params[0] is param_in_parent
    assert params[1] is param_in_child


def test_module_repr(parent_module):
    param = Parameter([1, 2, 3])
    parent_module.param = param

    module_repr = repr(parent_module)
    assert 'param' in module_repr
    assert 'child_module' in module_repr


def test_no_grad():
    a = Tensor([10, 20], requires_grad=True)
    with no_grad():
        b = a + 10
    assert b.context is None

