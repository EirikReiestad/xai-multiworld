import numpy as np
from numpy.typing import NDArray as ndarray

from swarm.utils.enum import IndexedEnum

OBJECT_SIZE = 8

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255]),
}

DIR_TO_VEC = [
    np.array((1, -1)),  # East
    np.array((1, 1)),  # Southeast
    np.array((0, 1)),  # South
    np.array((-1, 1)),  # Southwest
    np.array((-1, 0)),  # West
    np.array((-1, -1)),  # Northwest
    np.array((0, -1)),  # North
    np.array((1, -1)),  # Northeast
]


class WorldObjectType(str, IndexedEnum):
    unseen = "unseen"
    empty = "empty"
    circle = "circle"
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
    white = "white"

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


class State(str, IndexedEnum):
    """
    Enumeration of object states.
    """

    empty = "empty"
    contained = "contained"
    open = "open"
    closed = "closed"
    locked = "locked"
