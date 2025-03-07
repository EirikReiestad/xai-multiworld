import numpy as np

from multiworld.utils.enum import IndexedEnum

COLORS = {
    "DarkRed": np.array([238, 125, 95]),
    "DarkRedBorder": np.array([202, 65, 35]),
    "Yellow": np.array([255, 201, 67]),
    "LightYellow": np.array([255, 236, 189]),
    "LightGray": np.array([217, 217, 217]),
    "DarkGray": np.array([117, 117, 117]),
    "LightGreen": np.array([213, 243, 214]),
    "Green": np.array([131, 211, 127]),
    "GreenBorder": np.array([88, 153, 84]),
    "BlueLightBorder": np.array([94, 171, 248]),
    "LightBlue": np.array([201, 228, 252]),
    "Blue": np.array([94, 171, 248]),
    "Black": np.array([40, 45, 50]),
    "White": np.array([255, 255, 255]),
    "LightPink": np.array([246, 196, 233]),
    "Pink": np.array([229, 87, 189]),
    "LightRed": np.array([247, 207, 196]),
}


class Color(str, IndexedEnum):
    dark_red = "DarkRed"
    dark_red_border = "DarkRedBorder"
    yellow = "Yellow"
    light_yellow = "LightYellow"
    light_gray = "LightGray"
    dark_gray = "DarkGray"
    light_green = "LightGreen"
    green = "Green"
    green_border = "GreenBorder"
    blue_light_border = "BlueLightBorder"
    light_blue = "LightBlue"
    blue = "Blue"
    black = "Black"
    white = "White"
    light_pink = "LightPink"
    pink = "Pink"
    light_red = "LightRed"

    @staticmethod
    def cycle(n: np.int_):
        """
        Cycle through the available colors.
        """
        return tuple(Color.from_index(i % len("Color")) for i in range(int(n)))

    def rgb(self) -> np.ndarray[np.uint8]:
        """
        Return the RGB value of this ``Color``.
        """
        return COLORS[self]
