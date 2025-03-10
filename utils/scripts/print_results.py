import argparse
import json
import logging
import os
from typing import Dict, Tuple

import pandas as pd
from joblib.pool import np
from pandas.io.formats.style import Styler

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
        filepath = os.path.join(path, filename)

        with open(filepath, "r") as f:
            data = json.load(f)
        df, latex = tabulate_data(data, filename)
        if df is None:
            return
        latex_filename = filename.replace(".json", ".tex")
        latex_filepath = os.path.join(path, latex_filename)
        with open(latex_filepath, "w") as f:
            f.write(latex)
    else:
        for filename in os.listdir(path):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(path, filename), "r") as f:
                data = json.load(f)
                df, latex = tabulate_data(data, filename)
                if df is None:
                    break
                latex_filename = filename.replace(".json", ".tex")
                latex_filepath = os.path.join(path, latex_filename)
                with open(latex_filepath, "w") as f:
                    f.write(latex)


def tabulate_data(data: Dict, filename: str):
    if filename == "concept_scores.json":
        df = tabulate_generic(data)
        for concept, value in data.items():
            plot_3d(
                value,
                label=concept,
                min=0,
                max=1,
                show=False,
            )
    elif filename == "tcav_scores.json":
        df = tabulate_generic(data)
        for concept, value in data.items():
            plot_3d(
                value,
                label=concept,
                min=0,
                max=1,
                show=False,
            )

    files = {
        "cav_similarity.json": tabulate_similarity_matrix,
        "concept_combination_accuracies.json": tabulate_concept_combination_accuracies,
        "probe_robustness.json": tabulate_robustness,
        "probe_statistics.json": tabulate_probe_statistics,
        "cav_statistics.json": tabulate_probe_statistics,
        "sample_efficiency.json": tabulate_sample_efficiency,
    }

    df, latex = files[filename](data)

    logging.info("\n\n" + filename)
    logging.info(f"\n{df}")

    return df, latex


def tabulate_probe_statistics(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data)
    df.columns = df.columns.str.replace("_", r"\_")
    df.index = df.index.str.replace("_", r"\_")
    styled_df = df.style
    styled_df.apply(
        lambda s: highlight_max(s, props="color:{f_green}; bfseries:;"),
        axis=1,
    )
    styled_df.apply(
        lambda s: highlight_min(s, props="color:{f_darkred}; bfseries:;"),
        axis=1,
    )
    latex = styled_df.to_latex()
    latex = format_table(latex)
    return df, latex


def tabulate_similarity_matrix(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data)
    styled_df = df.style
    styled_df.apply(
        lambda s: highlight_between_threshold(
            s, props="color:{f_green}; bfseries:;", min=0.7, max=1.0
        ),
        axis=1,
    )
    latex = styled_df.to_latex()
    latex = format_table(latex)
    return df, latex


def tabulate_sample_efficiency(data: Dict) -> Tuple[pd.DataFrame, str]:
    pass


def tabulate_concept_combination_accuracies(data: Dict) -> Tuple[pd.DataFrame, str]:
    pass


def tabulate_robustness(data: Dict) -> Tuple[pd.DataFrame, str]:
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


def format_table(table: str) -> str:
    df_lines = table.splitlines()
    df_lines.insert(1, r"\toprule")
    df_lines.insert(3, r"\midrule")
    df_lines.insert(-1, r"\bottomrule")
    new_table = "\n".join(df_lines)
    return new_table


def highlight_max(s, props: str):
    return np.where(s == np.nanmax(s.values), props, "")


def highlight_min(s, props: str):
    return np.where(s == np.nanmin(s.values), props, "")


def highlight_threshold(s, props: str, min: float = -np.inf, max: float = np.inf):
    mask = (s >= min) & (s <= max)
    return [props if v else "" for v in mask]


def highlight_between_threshold(
    s, props: str, min: float = -np.inf, max: float = np.inf
):
    mask = (s > min) & (s < max)
    return [props if v else "" for v in mask]


if __name__ == "__main__":
    path = os.path.join("pipeline", "20250310-160946", "results")
    main(path)
