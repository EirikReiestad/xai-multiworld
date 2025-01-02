import aenum as enum
import numpy as np
from numpy.typing import NDArray as ndarray

from multigrid.utils.enum import IndexedEnum

TILE_PIXELS = 64

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
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


class WorldObjectType(str, IndexedEnum):
    unseen = "unseen"
    empty = "empty"
    wall = "wall"
    floor = "floor"
    box = "box"
    goal = "goal"
    agent = "agent"
    container = "container"


class Color(str, IndexedEnum):
    red = "red"
    green = "green"
    blue = "blue"
    purple = "purple"
    yellow = "yellow"
    grey = "grey"
    black = "black"

    @staticmethod
    def cycle(n: np.int_):
        """
        Cycle through the available colors.
        """
        return tuple(Color.from_index(i % len("Color")) for i in range(int(n)))

    def rgb(self) -> ndarray[np.uint8]:
        """
        Return the RGB value of this ``Color``.
        """
        return COLORS[self]


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

    # TODO: Find out why I can not place contained below empty, as every new object start with contained as their state, which is wrong.
    contained = "contained"
    empty = "empty"
    open = "open"
    closed = "closed"
    locked = "locked"
