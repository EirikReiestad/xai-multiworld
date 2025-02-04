import aenum as enum
import numpy as np
from numpy.typing import NDArray as ndarray

from multiworld.utils.enum import IndexedEnum

TILE_PIXELS = 64

DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


class WorldObjectType(str, IndexedEnum):
    unseen = "unseen"
    empty = "empty"
    goal = "goal"
    agent = "agent"
    wall = "wall"
    floor = "floor"
    box = "box"
    container = "container"


class Direction(enum.IntEnum):
    """
    Enumeration of agent directions.
    """

    right = 0
    down = 1
    left = 2
    up = 3

    def to_vec(self) -> ndarray[np.int8]:
        """
        Return the vector corresponding to this ``Direction``.
        """
        return DIR_TO_VEC[self]

    @staticmethod
    def sample(n: np.int_):
        """
        Sample a random direction.
        """
        return np.random.choice(list(Direction), size=n)


class State(str, IndexedEnum):
    """
    Enumeration of object states.
    """

    empty = "empty"
    contained = "contained"
    open = "open"
    closed = "closed"
    locked = "locked"
