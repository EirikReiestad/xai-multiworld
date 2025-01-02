from numpy.typing import NDArray
import numpy as np


def normalize_image(image: NDArray) -> NDArray:
    """
    Assuming image is shape (width, height, channels), normalize the image to be between 0 and 1.
    """
    min_value = np.nanmin(image)
    max_value = np.nanmax(image)

    normalized_image = (image - min_value) / (max_value - min_value)
    return normalized_image
