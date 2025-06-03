import os
import re
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from numpy.typing import NDArray

from experiments.src.constants import get_colormap

mpl.rcParams["axes3d.mouserotationstyle"] = (
    "azel"  # 'azel', 'trackball', 'sphere', or 'arcball'
)


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
        cmap=get_colormap(),
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
    X: dict[str, dict[str, float]],
    filename: str = "plot3d",
    folder_path: str = "assets/figures",
    label: str = "",
    title: str = "Plot",
    show: bool = True,
    min: float | None = None,
    max: float | None = None,
):
    X = {
        key: X[key]
        for key in sorted(X.keys(), key=lambda x: int(re.findall(r"_(\d+):", x)[-1]))
    }
    os.makedirs(folder_path, exist_ok=True)
    save_path = f"{folder_path}/{filename}.png"

    x_keys = list(next(iter(X.values())).keys())
    x_keys = sorted(x_keys, key=lambda x: int(x.split("-")[0]))
    y_keys = list(X.keys())

    matrix = np.array([[X[y][x] for x in x_keys] for y in y_keys])

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    _x = np.arange(matrix.shape[1])
    _y = np.arange(matrix.shape[0])
    _xx, _yy = np.meshgrid(_x, _y)

    min = matrix.min() if min is None else min
    max = matrix.max() if max is None else max

    ax.plot_surface(
        _xx,
        _yy,
        matrix,
        edgecolor="k",
        cmap=get_colormap(),
        alpha=1.0,
        shade=True,
    )

    label_fontsize = 18
    labelpad = 10
    ax.set_title(f"{title} - {label}", fontsize=26)
    ax.set_xlabel("Layer", fontsize=label_fontsize, labelpad=labelpad)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([x.split("-")[0] for x in x_keys])

    ax.set_ylabel("Checkpoint", fontsize=label_fontsize, labelpad=labelpad)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([re.findall(r"_(\d+):", y)[-1] for y in y_keys])

    ax.set_zlabel("Score", fontsize=label_fontsize, labelpad=labelpad)
    ax.set_zlim(min, max)
    ax.tick_params(labelsize=18)

    if show:
        plt.show()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
