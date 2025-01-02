from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt


def show_image(image: NDArray, title: str):
    image[np.isnan(image)] = 0

    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    plt.imshow(image)
    plt.title(title)
    plt.show()
