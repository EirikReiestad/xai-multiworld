import enum
import numpy as np
from numpy.typing import NDArray as ndarray
from multigrid.utils.enum import IndexedEnum

TILE_PIXELS = 32

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "black": (0, 0, 0),
}

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


class Type(str, IndexedEnum):
    unseen = "unseen"
    empty = "empty"
    wall = "wall"
    floor = "floor"
    box = "box"
    goal = "goal"
    agent = "agent"


class Color(str, IndexedEnum):
    red = "red"
    green = "green"
    black = "black"

    @staticmethod
    def cycle(n: np.int_):
        """
        Cycle through the available colors.
        """
        return tuple(Color.from_index(i % len(Color)) for i in range(int(n)))


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
