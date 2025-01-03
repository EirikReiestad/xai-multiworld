from numpy.typing import NDArray
import numpy as np


def remove_nan(arr: NDArray) -> NDArray:
    nan_arr = []
    for a in arr:
        if isinstance(a, np.ndarray) or not np.isnan(a):
            nan_arr.append(a)
    return np.array(nan_arr)
