import numpy as np
from multigrid.core.constants import Direction


def ohe_direction(direction: int) -> np.ndarray:
    result = np.array([direction == d for d in Direction], dtype=np.float32)
    return result
