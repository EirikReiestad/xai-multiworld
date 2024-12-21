import numpy as np


class Observation(np.ndarray):
    ID = 0
    LABEL = 1
    DATA = slice(2, None)

    dim = 2 + 1

    def __new__(cls, *dims: int):
        obj = np.empty(dims + (cls.dim,), dtype=object).view(cls)
        obj[..., cls.ID] = None
        obj[..., cls.LABEL] = None
        obj[..., cls.DATA] = None
