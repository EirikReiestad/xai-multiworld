import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_3d(
    X: Dict[str, Dict[str, Any]],
    filename: str = "plot3d.png",
    folder_path: str = "assets/figures",
    label: str = "",
    title: str = "Plot",
    show: bool = True,
    min: float | None = None,
    max: float | None = None,
):
    os.makedirs(folder_path, exist_ok=True)
    save_path = f"{folder_path}/{filename}"

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

    fig.tight_layout()

    if show:
        plt.show()
    fig.savefig(save_path)
    plt.close(fig)
