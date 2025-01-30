import numpy as np
from numpy.typing import NDArray as ndarray

from multiworld.utils.enum import IndexedEnum

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255]),
}


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
