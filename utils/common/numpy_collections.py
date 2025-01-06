from numpy.typing import NDArray
import numpy as np
import json
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


def dict_list_to_ndarray(dict_list: list[dict[str, Any]]) -> NDArray:
    data = []
    for key in dict_list[0].keys():
        data.append(np.array([d[key] for d in dict_list]))
    return np.stack(data)
