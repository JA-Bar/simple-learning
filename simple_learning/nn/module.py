from .parameter import Parameter


class Module:
    def __init__(self):
        # enforce the initialization of Module on subclasses
        self._parameters = {}
        self._modules = {}
        self._attributes = set()

    def forward(self):
        raise NotImplementedError

    def get_named_parameters(self):
        all_named_parameters = {}

        # inner function to recurse into the submodules
        def _log_params(module, params, param_prefix):
            # update the params dictionary with the module's parameters
            for name, value in module._parameters.items():
                params.update({param_prefix + name: value})

            # add to the param_prefix the name of this module, then recurse
            for name, sub_module in module._modules.items():
                sub_param_prefix = param_prefix + name + '.'
                _log_params(sub_module, params, sub_param_prefix)

        # initialize the param_prefix with the top module's name
        param_prefix = self.__module__ + '.' + self.__class__.__name__ + '.'
        _log_params(self, all_named_parameters, param_prefix)

        return all_named_parameters

    def get_parameters(self):
        return list(self.get_named_parameters().values())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, *kwargs)

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]
        else:
            raise AttributeError(f"'{name}' not found in '{self.__class__.__name__}'")

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters.update({name: value})
        elif isinstance(value, Module):
            self._modules.update({name: value})
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        _repr = f"Module: {self.__class__.__name__}"
        if len(self._parameters) > 0:
            _repr += ' | Parameters: ' + ', '.join(self._parameters.keys())
        if len(self._modules) > 0:
            _repr += ' | Child Modules: ' + ', '.join(self._modules.keys())
        return _repr

