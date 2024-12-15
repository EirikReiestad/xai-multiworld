from multigrid.core.world_object import WorldObject
from multigrid.utils.position import Position
from multigrid.core.grid import Grid
from typing import Callable
import numpy as np


class Area(np.ndarray):
    """
    Area consisting of multiple objects arranged in a square shape.
    All objects are of the same type.
    """

    def __new__(cls, shape: tuple[int, int], tile_generator: Callable[[], WorldObject]):
        """
        Parameters
        ----------
        shape : tuple[int, int]
            Shape of the grid (height, width) for the area.
        tile_generator : Callable[[], WorldObject]
            Function to generate each WorldObject tile.
        """
        obj = np.empty(shape, dtype=object).view(cls)

        for i in range(shape[0]):
            for j in range(shape[1]):
                obj[i, j] = tile_generator()

        return obj

    def place(self, grid: Grid, pos: tuple[int, int]):
        """
        Place the area at the given position in the grid.
        Parameters
        ----------
        grid : np.ndarray
            Grid to place the area in.
        pos : tuple[int, int]
            Position to place the area at.
        """
        h, w = self.shape
        for i in range(h):
            for j in range(w):
                position = Position(pos[0] + i, pos[1] + j)
                if not grid.in_bounds(position):
                    continue
                grid.set(position, self[i, j])
