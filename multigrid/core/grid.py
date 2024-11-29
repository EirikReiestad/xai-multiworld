from multigrid.core.world_object import WorldObject
import numpy as np
from numpy.typing import NDArray


class Grid:
    _tile_cache = {}

    def __init__(self, width: int = 10, height: int = 10):
        assert width >= 3
        assert height >= 3

        self._world_objects: dict[
            tuple[int, int], WorldObject
        ] = {}  # index by position
        self.state: NDArray[np.int32] = np.zeros(
            (width, height, WorldObject.dim), dtype=int
        )
        self.state[...] = WorldObject.empty()
