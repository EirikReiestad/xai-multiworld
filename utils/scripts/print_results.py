import argparse
import json
import logging
import os
from typing import Dict, Tuple

import pandas as pd
from joblib.pool import np

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
        if df is None or latex is None:
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
                if df is None or latex is None:
                    continue
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
        "concept_cav_similarity.json": tabulate_similarity_matrix,
        "cos_sim_matrix.json": tabulate_similarity_matrix,
        "cav_statistics.json": tabulate_cav_statistics,
        "probe_statistics.json": tabulate_cav_statistics,
        "concept_cav_statistics.json": tabulate_cav_statistics,
        "cav_training_stats.json": tabulate_training_stats,
        "concept_combination_accuracies.json": tabulate_combination_accuracies,
        "probe_robustness.json": tabulate_robustness,
        "sample_efficiency.json": tabulate_sample_efficiency,
        "concept_score_decision_tree.json": tabulate_concept_score_decision_tree,
        "shapley_values.json": tabulate_shapley_values,
    }

    try:
        df, latex = files[filename](data)
    except KeyError:
        logging.error(f"File {filename} not found in tabulate_data")
        return None, None

    logging.info("\n\n" + filename)
    logging.info(f"\n{df}")

    return df, latex


def tabulate_generic(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data)
    latex = df.to_latex()
    return df, latex


def tabulate_shapley_values(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame([data])
    df = df.transpose()
    df.columns = ["Shapley Value"]
    df.sort_values(by="Shapley Value", ascending=False, inplace=True)
    styled_df = df.copy()
    styled_df = styled_df.style
    latex = styled_df.to_latex()
    latex = format_table(latex)
    return df, latex


def tabulate_robustness(data: Dict) -> Tuple[pd.DataFrame, str]:
    processed_data = {}
    for condition, values in data.items():
        if condition not in processed_data:
            processed_data[condition] = {}
        for key, value in values.items():
            processed_data[condition][key] = value[0][0]
    data = processed_data
    df = pd.DataFrame(data)
    styled_df = df.copy()
    styled_df.index = styled_df.index.str.replace("_", r"\_")
    styled_df.columns = styled_df.columns.str.replace("_", r"\_")
    styled_df = styled_df.style
    styled_df.apply(
        lambda s: highlight_threshold(s, props="bfseries:;", min=0.7, max=1.0),
        axis=0,
    )
    latex = styled_df.to_latex()
    latex = format_table(latex)
    return df, latex


def tabulate_concept_score_decision_tree(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = ["Importance", "Splits"]
    df.sort_values(by="Importance", ascending=False, inplace=True)
    styled_df = df.copy()
    styled_df.index = styled_df.index.str.replace("_", r"\_")
    styled_df = styled_df.style
    latex = styled_df.to_latex()
    latex = format_table(latex)
    return df, latex


def tabulate_combination_accuracies(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = ["Loss", "Accuracy"]
    df.sort_values(by="Loss", ascending=True, inplace=True)
    styled_df = df.copy()
    styled_df.index = styled_df.index.str.replace("_", r"\_")
    styled_df = styled_df.style
    styled_df.apply(
        lambda s: highlight_max(s, props="color:{f_green}; bfseries:;"),
        axis=0,
    )
    styled_df.apply(
        lambda s: highlight_min(s, props="color:{f_darkred}; bfseries:;"),
        axis=0,
    )
    latex = styled_df.to_latex()
    latex = format_table(latex)
    return df, latex


def tabulate_training_stats(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data, index=None)
    styled_df = df.copy()
    styled_df.columns = styled_df.columns.str.replace("_", r"\_")
    styled_df.index = styled_df.index.str.replace("_", r"\_")
    latex = styled_df.to_latex(index=False)
    latex = format_table(latex)
    return df, latex


def tabulate_cav_statistics(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data)
    styled_df = df.copy()
    styled_df.columns = styled_df.columns.str.replace("_", r"\_")
    styled_df.index = styled_df.index.str.replace("_", r"\_")
    styled_df = styled_df.style
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
    np_df = df.to_numpy()
    tril = np.tril(np_df)
    mask = np.triu(np.ones(tril.shape), k=1).astype(bool)
    tril[mask] = np.nan
    df = pd.DataFrame(tril, columns=df.columns, index=df.index)
    styled_df = df.copy()
    styled_df.columns = styled_df.columns.str.replace("_", r"\_")
    styled_df.index = styled_df.index.str.replace("_", r"\_")
    styled_df = styled_df.style
    styled_df.apply(
        lambda s: highlight_between_threshold(
            s, props="color:{f_green}; bfseries:;", min=0.7, max=1.0
        ),
        axis=1,
    )
    styled_df.apply(highlight_nan_values, props="color:{f_white}; bfseries:;", axis=1)
    latex = styled_df.to_latex()
    latex = format_table(latex)
    return df, latex


def tabulate_sample_efficiency(data: Dict) -> Tuple[pd.DataFrame, str]:
    pass


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


def highlight_nan_values(s, props: str):
    return np.where(pd.isna(s), props, "")


if __name__ == "__main__":
    path = os.path.join("pipeline", "20250310-160946", "results")
    main(path)
