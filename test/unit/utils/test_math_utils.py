# from test import get_resource_dir

import numpy as np

import ugvc.utils.math_utils as math_utils

# inputs_dir = get_resource_dir(__file__)


def test_phred():
    assert np.all(math_utils.phred((0.1, 0.01, 0.001)) == np.array([10.0, 20.0, 30.0]))


def test_unphred():
    assert np.all(math_utils.unphred((10, 20, 30)) == np.array([0.1, 0.01, 0.001]))
