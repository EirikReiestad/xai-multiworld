import json
import os

import matplotlib.pyplot as plt
from experiments.src.constants import get_palette


def main(
    filename: str,
    folder: str = "experiments/results",
    title: str = "Layer Values for Different Concepts",
    savefig: str = "experiments/plots/scores.png",
):
    with open(os.path.join(folder, filename)) as f:
        data = json.load(f)

    layers = list(data["0"]["latest"].keys())
    concepts = list(data.keys())
    values = {
        concept: [data[concept]["latest"][layer] for layer in layers][::-1]
        for concept in concepts
    }
    layers = [str(i) for i in range(len(layers))]

    plt.figure(figsize=(12, 6))
    palette = get_palette([str(d) for d, _ in values.items()])
    for concept, y in values.items():
        plt.plot(
            layers,
            y,
            color=palette[concept],
            marker="o",
            label=f"Concept {concept}",
            markersize=10,
            linewidth=2.5,
        )

    plt.title(title)
    plt.xlabel("Layers")
    plt.ylabel("Scores")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=4,
        frameon=False,
    )
    plt.grid()
    plt.tight_layout()
    plt.savefig(savefig)
    # plt.show()


if __name__ == "__main__":
    main(
        "concept_scores.json",
        title="Concept scores for different concepts",
        savefig="experiments/plots/concept_scores.png",
    )
    main(
        "tcav_scores.json",
        title="TCAV scores for different concepts",
        savefig="experiments/plots/tcav_scores.png",
    )
