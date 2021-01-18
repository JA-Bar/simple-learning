import numpy as np

import torch.nn.functional as torch_F

import simple_learning.functional as sl_F

from helper_commons import evaluate_function_with_pytorch


np.random.seed(42)

# absolute and relative tolerances
A_TOLERANCE = 1e-6
R_TOLERANCE = 1e-6


class TestNNFunctions:
    scenario_1 = ('scalar', {'constructor': [np.random.randn, (1, )]})
    scenario_2 = ('one_dimensional', {'constructor': [np.random.randn, (5, )]})
    scenario_3 = ('row_vector', {'constructor': [np.random.randn, (1, 5)]})
    scenario_4 = ('multiple_vectors', {'constructor': [np.random.randn, (10, 5)]})

    scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

    def test_relu(self, constructor):
        evaluate_function_with_pytorch(sl_F.relu, torch_F.relu, constructor)

    def test_softmax(self, constructor):
        # adding extra pow function to make the softmax derivatives bigger
        sl_function = lambda x: (sl_F.softmax(x)**4.2).sum()
        torch_function = lambda x: (torch_F.softmax(x, dim=-1)**4.2).sum()
        evaluate_function_with_pytorch(sl_function, torch_function, constructor)


