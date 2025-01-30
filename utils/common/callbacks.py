from typing import Callable, Dict, Optional
import numpy as np

from numpy.typing import NDArray

from multigrid.utils.typing import ObsType
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image as PILImage

RenderingCallback = Callable[[NDArray, Optional[ObsType]], NDArray]


def empty_rendering_callback(
    image: NDArray, observations: Optional[Dict[str, ObsType]] = None
) -> NDArray:
    return image


def kernel_density_estimation_callback(
    image: NDArray, observations: Optional[Dict[str, ObsType]] = None
) -> NDArray:
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    y, x = np.where(gray_image > 0)

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))

    sns.kdeplot(
        x=x,
        y=y,
        cmap="mako",
        fill=True,
        thresh=0,
        levels=100,
        alpha=0.1,
        ax=ax,
        bw_method=0.1,
        # bw_adjust=1,
    )

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    heatmap_image = PILImage.open(buf)

    heatmap_image = heatmap_image.resize(
        (image.shape[1], image.shape[0]), PILImage.Resampling.LANCZOS
    )
    heatmap_image = np.array(heatmap_image)
    heatmap_image = np.flip(heatmap_image, axis=0)

    heatmap_alpha = heatmap_image[..., 3] / 255.0

    blended_image = (
        image.astype(np.float32) * (1 - heatmap_alpha[..., None] * 0.4)
        + heatmap_image[..., :3].astype(np.float32) * heatmap_alpha[..., None] * 0.4
    )

    # blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    return blended_image
