import json
import os
from itertools import count

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.core.constants import Color, get_palette


def read_accuracy(directory: str, base_filename: str):
    accuracies = []
    base = os.path.join(directory, base_filename)
    for i in count():
        try:
            with open(f"{base}_{i}.json") as f:
                data = json.load(f)
        except FileNotFoundError:
            break
        accuracies.append(data["accuracy"])
    return accuracies


def main(directory: str, base_filename: str):
    base_human = base_filename + "_human"
    accuracies = read_accuracy(directory, base_filename)
    accuracies_human = read_accuracy(directory, base_human)

    data = {
        "human-defined concepts": accuracies_human,
        "automatically discovered concepts": accuracies,
    }
    df = pd.DataFrame(data)
    df_melt = df.melt(var_name="method", value_name="score")

    labels = list(data.keys())
    palette = get_palette(labels + ["", ""])
    colors = [palette[i] for i in labels]

    df = pd.DataFrame(data)
    df_melt = df.melt(var_name="method", value_name="score")

    labels = list(data.keys())
    palette = get_palette(labels)
    colors = [palette[i] for i in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x="method", y="score", ax=ax, data=df_melt, hue="method", palette=palette
    )
    ax.axhline(
        0.42,
        color=Color(Color.dark_red),
        linestyle="--",
        linewidth=2,
        label="Baseline: 0.42",
    )
    ax.legend(fontsize=16)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=18)  # increase x tick labels
    ax.set_ylabel("Accuracy", fontsize=18)  # increase y label
    ax.set_xlabel("Method", fontsize=18)  # set x label and size if needed
    ax.tick_params(axis="y", labelsize=16)  # increase y tick labels
    ax.set_title("Decision tree accuracy", fontsize=22)  # larger title
    plt.tight_layout()
    plt.show()
    plt.savefig("assets/figures/classification_report_decision_tree.png")


if __name__ == "__main__":
    main(
        directory="assets/resultsv0/",
        base_filename="classification_report_decision_tree",
    )
