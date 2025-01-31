import numpy as np

from multiworld.utils.enum import IndexedEnum

OBJECT_SIZE = 8

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


class State(str, IndexedEnum):
    """
    Enumeration of object states.
    """

    empty = "empty"
    contained = "contained"
    open = "open"
    closed = "closed"
    locked = "locked"
