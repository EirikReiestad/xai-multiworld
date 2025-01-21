from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from typing import List
import matplotlib.colors as mcolors
import seaborn as sns
import os
from typing import Dict, Any
from matplotlib import cm


def box_plot(
    data: List,
    title: str,
    labels=["red", "green", "blue"],
    colors=["red", "green", "blue"],
    x_label="",
    y_label="",
):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    bplot = ax.boxplot(
        data,
        patch_artist=True,  # fill with color
        tick_labels=labels,
    )  # will be used to label x-ticks

    # fill with colors
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    plt.title(title)
    plt.show()


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
        fig, axes = plt.subplots(1, 3, figsize=(15, 10))

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

    plt.close()


def plot_3d(
    X: Dict[str, Dict[str, Any]],
    filename: str = "plot3d",
    folder_path: str = "assets/figures",
    label: str = "",
    title: str = "Plot",
    show: bool = True,
    min: float | None = None,
    max: float | None = None,
):
    os.makedirs(folder_path, exist_ok=True)
    save_path = f"{folder_path}/{filename}.png"

    x_keys = X.keys()
    y_keys = next(iter(X.values())).keys()
    matrix = np.array([[X[x][y] for y in y_keys] for x in x_keys])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    _x = np.arange(matrix.shape[1])
    _y = np.arange(matrix.shape[0])
    _xx, _yy = np.meshgrid(_x, _y)

    min = matrix.min() if min is None else min
    max = matrix.max() if max is None else max

    cmap = cm.get_cmap("plasma")
    ax.plot_surface(
        _xx,
        _yy,
        matrix,
        edgecolor="k",
        cmap=cmap,
        alpha=1.0,
        shade=True,
    )

    ax.set_title(f"{title} - {label}", fontsize=16)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Steps", fontsize=14)
    ax.set_zlabel("Score", fontsize=14)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(y_keys, rotation=45)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(x_keys)
    ax.set_zlim(min, max)

    if show:
        plt.show()
    fig.savefig(save_path)
    plt.close(fig)
