import argparse
import json
import logging
import os
from typing import Dict

import pandas as pd

from utils.core.plotting import plot_3d

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(path: str = os.path.join("assets", "results")):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pr",
        "--print-results",
        nargs="*",
        metavar=("filename"),
        help="Render the observations stored in json files under assets/concepts with optional arguement [filename]",
    )

    filename = None
    args = parser.parse_args()
    if args.print_results is not None and len(args.print_results) > 0:
        filename = args.print_results[0]

    if filename is not None:
        filename = filename + ".json" if not filename.endswith(".json") else filename
        path = os.path.join(path, filename)

        with open(path, "r") as f:
            data = json.load(f)
        tabulate_data(data, filename)
    else:
        for filename in os.listdir(path):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(path, filename), "r") as f:
                data = json.load(f)
                tabulate_data(data, filename)


def tabulate_data(data: Dict, filename: str):
    if filename == "concept_scores.json":
        df = tabulate_generic(data)
        for concept, value in data.items():
            plot_3d(
                value,
                label=concept,
                min=0,
                max=1,
                show=True,
            )
    elif filename == "tcav_scores.json":
        df = tabulate_generic(data)
        for concept, value in data.items():
            plot_3d(
                value,
                label=concept,
                min=0,
                max=1,
                show=True,
            )
    elif filename == "cos_sim_matrix.json":
        df = pd.DataFrame(data)
    elif filename.startswith("concept_combination_accuracies"):
        df = tabulate_combination_accuracy(data)
        df = df.sort_values(by="Count", ascending=False)
    elif filename == "probe_robustness.json":
        df = tabulate_robustness(data)
    elif filename == "probe_statistics.json":
        df = pd.DataFrame(data)
    elif filename == "sample_efficiency.json":
        df = pd.DataFrame(data).transpose()
    else:
        logging.warning(f"Value {filename} not supported")
        return
    print("\n\n" + filename)
    print(df)


def tabulate_generic(data: Dict):
    return pd.DataFrame(
        [
            {"Condition": condition, "Model": model, **values}
            for condition, models in data.items()
            for model, values in models.items()
        ]
    )


def tabulate_robustness(data: Dict):
    df = pd.DataFrame(
        [
            {"Condition": condition, "Key": key, "Value": value[0][0]}
            for condition, values in data.items()
            for key, value in values.items()
        ]
    )
    return df.pivot(index="Key", columns="Condition", values="Value").iloc[::-1]


def tabulate_combination_accuracy(data: Dict):
    return pd.DataFrame.from_dict(data, orient="index", columns=["Value", "Count"])


if __name__ == "__main__":
    path = os.path.join("pipeline", "20250310-160946", "results")
    main(path)
