import numpy as np


class HashableArray:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __hash__(self):
        return hash(self.array.tobytes())

    def __eq__(self, other):
        return isinstance(other, HashableArray) and np.array_equal(
            self.array, other.array
        )
