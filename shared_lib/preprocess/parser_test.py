import numpy as np

from shared_lib.preprocess import parser as parser


def test_normalize_planes():
    arr = np.array([-3, 0, 4, 7, 9, 10, 14, 54])
    norm = parser.normalize_planes(arr, 0.0, 10.0)
    assert np.array_equal(norm, [0, 0, 0.4, 0.7, 0.9, 1, 1, 1])
