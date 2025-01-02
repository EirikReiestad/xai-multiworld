from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray


def plot_heatmap(
    data: NDArray,
    background: NDArray | None = None,
    alpha: float = 0.5,
    title: str = "",
    yticks: List | bool = False,
    xticks: List | bool = False,
    cmap="coolwarm",
    figsize=(10, 8),
    cbar=True,
    show: bool = False,
    save: bool = False,
):
    plt.figure(figsize=figsize)

    if background is not None:
        plt.imshow(
            background[::-1],
            aspect="auto",
            extent=(0, data.shape[1], 0, data.shape[0]),
        )

    norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())

    sns.heatmap(
        data,
        annot=True,
        xticklabels=xticks,
        yticklabels=yticks,
        cmap=cmap,
        cbar=cbar,
        alpha=alpha,
        norm=norm,
    )
    plt.axis("off")
    plt.title(title)

    if show:
        plt.show()

    if save:
        plt.savefig(f"assets/figures/heatmap-{title}.png")
