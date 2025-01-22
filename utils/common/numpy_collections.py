import numpy as np
import json
import torch
from numpy.typing import NDArray
from typing import Any


class HashableArray:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __hash__(self):
        return hash(self.array.tobytes())

    def __eq__(self, other):
        return isinstance(other, HashableArray) and np.array_equal(
            self.array, other.array
        )


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def remove_nan(arr: NDArray) -> NDArray:
    nan_arr = []
    for a in arr:
        if isinstance(a, np.ndarray) or not np.isnan(a):
            nan_arr.append(a)
    return np.array(nan_arr)


def normalize_ndarrays(arr: NDArray, a: float = 0, b: float = 1) -> NDArray:
    global_min = np.min(arr)
    global_max = np.max(arr)

    states = (arr - global_min) / (global_max - global_min) * (b - a) + a

    return states
