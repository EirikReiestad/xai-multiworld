from typing import List

import numpy as np
from PIL import Image


def save_gif(frames: List[np.ndarray], file_path: str):
    """
    Convert a list of frames to a gif file.
    """
    pil_images = [Image.fromarray(frame.transpose(1, 0, 2), "RGB") for frame in frames]

    assert len(pil_images) > 0, "No frames to save."
    assert all(
        isinstance(image, Image.Image) for image in pil_images
    ), f"Images must be of type PIL.Image.Image, but got {pil_images}"

    pil_images[0].save(
        file_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=100,  # Duration between frames in ms
        loop=0,  # Loop forever
    )
