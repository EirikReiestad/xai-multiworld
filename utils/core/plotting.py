from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def show_image(
    image: NDArray,
    title: str,
    rgb: bool = False,
    rgb_titles: Tuple = ("red", "green", "blue"),
):
    image[np.isnan(image)] = 0

    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    if rgb is True:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, ax in enumerate(axes):
            channel_image = image.copy()
            for j in range(3):
                if j != i:
                    channel_image[:, :, j] = 0
            ax.imshow(channel_image)
            ax.set_title(f"{rgb_titles[i]} Channel")
            ax.axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        return

    plt.imshow(image)
    plt.title(title)
    plt.show()
