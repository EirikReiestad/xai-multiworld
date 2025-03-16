from joblib.pool import np

from multiworld.multigrid.core.constants import Direction
from utils.core.constants import Color


def add_highlighted_border(
    image: np.ndarray, direction: int, border_size: int = 10
) -> np.ndarray:
    border_color = Color(Color.light_blue).rgb()
    background_color = Color(Color.black).rgb()

    height, width, channels = image.shape

    new_height = height + 2 * border_size
    new_width = width + 2 * border_size

    image_with_borders = np.full(
        (new_height, new_width, channels), background_color, dtype=np.uint8
    )

    # Place the original image in the center of the new image
    image_with_borders[
        border_size : border_size + height, border_size : border_size + width
    ] = image

    # Highlight the border based on the direction
    if direction == Direction.right:
        image_with_borders[
            border_size : border_size + height, new_width - border_size :
        ] = border_color
    elif direction == Direction.down:
        image_with_borders[
            new_height - border_size :, border_size : border_size + width
        ] = border_color
    elif direction == Direction.left:
        image_with_borders[border_size : border_size + height, :border_size] = (
            border_color
        )
    elif direction == Direction.up:
        image_with_borders[:border_size, border_size : border_size + width] = (
            border_color
        )
    else:
        raise ValueError("Direction must be valid:)")

    return image_with_borders
