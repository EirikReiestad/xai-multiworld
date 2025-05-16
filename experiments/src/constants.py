from typing import List
import matplotlib.colors as mcolors
import numpy as np

from src.utils import IndexedEnum

COLORS = {
    "Gray": np.array([169, 169, 169]),
    "LightGray": np.array([217, 217, 217]),
    "DarkGray": np.array([117, 117, 117]),
    "DarkRed": np.array([238, 125, 95]),
    "LightRed": np.array([247, 207, 196]),
    "DarkRedBorder": np.array([202, 65, 35]),
    "Yellow": np.array([255, 201, 67]),
    "LightYellow": np.array([255, 236, 189]),
    "LightGreen": np.array([213, 243, 214]),
    "Green": np.array([131, 211, 127]),
    "GreenBorder": np.array([88, 153, 84]),
    "LightBlue": np.array([201, 228, 252]),
    "LightBlueBorder": np.array([94, 171, 248]),
    "Blue": np.array([94, 171, 248]),
    "Black": np.array([40, 45, 50]),
    "White": np.array([255, 255, 255]),
    "LightPink": np.array([246, 196, 233]),
    "Pink": np.array([229, 87, 189]),
}


class Color(str, IndexedEnum):
    gray = "Gray"
    light_gray = "LightGray"
    dark_gray = "DarkGray"
    dark_red = "DarkRed"
    light_red = "LightRed"
    dark_red_border = "DarkRedBorder"
    yellow = "Yellow"
    light_yellow = "LightYellow"
    light_green = "LightGreen"
    green = "Green"
    green_border = "GreenBorder"
    light_blue_border = "LightBlueBorder"
    light_blue = "LightBlue"
    blue = "Blue"
    black = "Black"
    white = "White"
    light_pink = "LightPink"
    pink = "Pink"

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


def get_colormap():
    color_sequence_names = ["LightBlue", "Green", "Yellow", "DarkRed"]
    normalized_colors = [COLORS[name] / 255.0 for name in color_sequence_names]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "my_custom_map", normalized_colors
    )
    return custom_cmap


def get_palette(categories: List[str]) -> dict:
    cmap = get_colormap()  # Get the custom colormap instance
    n_categories = len(categories)

    # Generate colors by sampling the colormap at equidistant points
    colors = (
        [cmap(i / (n_categories - 1)) for i in range(n_categories)]
        if n_categories > 1
        else [cmap(0.5)]
    )

    colormap_palette = {category: colors[i] for i, category in enumerate(categories)}
    return colormap_palette
