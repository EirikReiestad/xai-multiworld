import argparse
import json
from utils.core.constants import get_colormap, get_palette
import logging
import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from joblib.pool import np

from utils.core.plotting import plot_3d

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


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
                figure_filename = filename.replace(".json", ".png")
                figure_filepath = os.path.join(path, figure_filename)
                plot_data(df, filename, figure_filepath)


def tabulate_data(data: Dict, filename: str, verbose: bool = False):
    files = {
        "concept_scores.json": tabulate_generic,
        "tcav_scores.json": tabulate_generic,
        "cav_similarity.json": tabulate_similarity_matrix,
        "concept_cav_similarity.json": tabulate_similarity_matrix,
        "probe_similarities.json": tabulate_similarity_matrix,
        "cav_statistics.json": tabulate_cav_statistics,
        "probe_statistics.json": tabulate_cav_statistics,
        "concept_cav_statistics.json": tabulate_cav_statistics,
        "cav_training_stats.json": tabulate_training_stats,
        "probe_robustness.json": tabulate_robustness,
        "sample_efficiency.json": tabulate_sample_efficiency,
        "concept_completeness_score_decision_tree.json": tabulate_concept_score_decision_tree,
        "cav_completeness_score_decision_tree.json": tabulate_concept_score_decision_tree,
        "concept_completeness_score_network.json": tabulate_concept_score_network,
        "info_concept_completeness_score_decision_tree.json": tabulate_concept_score_decision_tree_info,
        "info_cav_completeness_score_decision_tree.json": tabulate_concept_score_decision_tree_info,
        "shapley_concept_completeness_score_network.json": tabulate_shapley_values,
    }

    try:
        df, latex = files[filename](data)
    except KeyError as e:
        if verbose:
            logging.error(f"File {filename} not found in tabulate_data: {e}")
        return None, None

    if verbose:
        logging.info("\n\n" + filename)
        logging.info(f"\n{df}")

    return df, latex


def plot_data(df: pd.DataFrame, filename: str, savepath: str, verbose: bool = True):
    files = {
        "cav_similarity.json": plot_similarity_matrix,
        "concept_cav_similarity.json": plot_similarity_matrix,
        "probe_similarities.json": plot_similarity_matrix,
        "cav_statistics.json": plot_cav_statistics,
        "probe_statistics.json": plot_cav_statistics,
        "concept_cav_statistics.json": plot_cav_statistics,
        "probe_robustness.json": plot_robustness,
        "sample_efficiency.json": plot_sample_efficiency,
        "concept_completeness_score_decision_tree.json": plot_concept_score_decision_tree,
        "cav_completeness_score_decision_tree.json": plot_concept_score_decision_tree,
    }
    try:
        files[filename](df)
        plt.tight_layout()
        plt.savefig(savepath, bbox_inches="tight")
    except KeyError as e:
        if verbose:
            logging.error(f"File {filename} not found in plot_data: {e}")
        return None, None


def tabulate_generic(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data)
    latex = df.to_latex()
    return df, latex


def plot_generic(df: pd.DataFrame):
    raise


def tabulate_concept_score_decision_tree_info(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame([data])
    df = df.transpose()
    df.columns = [""]
    styled_df = df.copy()
    styled_df.index = styled_df.index.str.replace("_", r"\_")
    styled_df = styled_df.style
    latex = styled_df.to_latex()
    latex = format_table(latex)
    return df, latex


def tabulate_shapley_values(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame([data])
    df = df.transpose()
    df.columns = ["Shapley Value"]
    df.sort_values(by="Shapley Value", ascending=False, inplace=True)
    styled_df = df.copy()
    styled_df.index = styled_df.index.str.replace("_", r"\_")
    styled_df = styled_df.style
    latex = styled_df.to_latex()
    latex = format_table(latex)
    return df, latex


def plot_shapley_values(df: pd.DataFrame):
    raise


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
        lambda s: highlight_threshold(s, props="bfseries:;", min=0.9, max=1.0),
        axis=0,
    )
    latex = styled_df.to_latex()
    latex = format_table(latex)
    return df, latex


def plot_robustness(df: pd.DataFrame):
    categories = df.columns
    df_reset = df.reset_index().rename(columns={"index": "threshold"})

    # Melt the DataFrame for plotting with seaborn
    df_melted = pd.melt(
        df_reset, id_vars="threshold", var_name="category", value_name="value"
    )

    color_palette = get_palette(list(categories))

    # Ensure threshold is numeric
    df_melted["threshold"] = pd.to_numeric(df_melted["threshold"])

    # Sort for proper line plotting
    df_melted = df_melted.sort_values(by=["category", "threshold"])

    # Create the line plot
    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=df_melted,
        x="threshold",
        y="value",
        hue="category",
        marker="o",
        palette=color_palette,
    )

    plt.title("Value vs. Threshold for Different Categories", fontsize=16)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.gca().invert_xaxis()  # Invert x-axis for thresholds


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


def plot_concept_score_decision_tree(df: pd.DataFrame):
    df_reset = df.reset_index().rename(columns={"index": "category"})

    categories = df_reset["category"].unique()
    colormap_palette = get_palette(list(categories))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_reset,
        x="category",
        y="Importance",
        hue="category",
        palette=colormap_palette,
        legend=False,
    )
    plt.title("Concept Importance", fontsize=16)
    plt.xlabel("Concept", fontsize=12)
    plt.ylabel("Importance Score", fontsize=12)
    plt.xticks(rotation=45, ha="right")


def tabulate_concept_score_network(data: Dict) -> Tuple[pd.DataFrame, str]:
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


def plot_concept_score_network(df: pd.DataFrame):
    pass


def tabulate_training_stats(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data, index=None)
    styled_df = df.copy()
    styled_df.columns = styled_df.columns.str.replace("_", r"\_")
    styled_df.index = styled_df.index.str.replace("_", r"\_")
    latex = styled_df.to_latex(index=False)
    return df, latex


def plot_training_stats(df: pd.DataFrame):
    raise


def tabulate_cav_statistics(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data)
    df.index.name = "statistic"
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


def plot_cav_statistics(df: pd.DataFrame):
    df_reset = df.reset_index()

    categories = ["density", "distance"]
    df_filtered = df_reset[df_reset["statistic"].isin(categories)]

    df_melted = pd.melt(
        df_filtered, id_vars="statistic", var_name="column_index", value_name="value"
    )

    colormap_palette = get_palette(categories)

    plt.figure(figsize=(18, 8))
    sns.barplot(
        data=df_melted,
        x="column_index",
        y="value",
        hue="statistic",
        palette=colormap_palette,
    )

    plt.title("Statistics per Column Index", fontsize=16)
    plt.xlabel("Column Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Statistic", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()  # Adjust layout to prevent labels overlapping


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


def plot_similarity_matrix(df: pd.DataFrame):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=False, fmt=".3f", cmap=get_colormap(), linewidths=0.5)
    # annot=True: show values in cells
    # fmt=".2f": format values to 3 decimal places
    # cmap="viridis": choose a color map (many options available, e.g., "coolwarm", "YlGnBu")
    # linewidths: add lines between cells

    plt.title("Data Matrix Heatmap", fontsize=16)
    plt.xlabel("Column Index", fontsize=12)
    plt.ylabel("Row Index", fontsize=12)
    plt.tight_layout()


def tabulate_sample_efficiency(data: Dict) -> Tuple[pd.DataFrame, str]:
    df = pd.DataFrame(data, index=None)
    df = df.transpose()
    df.sort_values(by="normalized", ascending=False, inplace=True)
    styled_df = df.copy()
    styled_df.columns = styled_df.columns.str.replace("_", r"\_")
    styled_df.index = styled_df.index.str.replace("_", r"\_")
    latex = styled_df.to_latex()
    return df, latex


def plot_sample_efficiency(df: pd.DataFrame):
    # Check if 'normalized' column exists
    if "normalized" not in df.columns:
        logging.error(
            "Error: DataFrame does not contain a 'normalized' column to plot."
        )
        return

    # Reset index to use categories as x-axis
    df_reset = df.reset_index().rename(columns={"index": "category"})

    categories = df_reset["category"].unique()
    colormap_palette = get_palette(list(categories))

    plt.figure(figsize=(12, 6))
    # Plot the 'normalized' column. df is already sorted.
    sns.barplot(
        data=df_reset,
        x="category",
        y="normalized",
        hue="category",
        palette=colormap_palette,
        legend=False,
    )

    plt.title("Sample Efficiency (Normalized)", fontsize=16)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.xticks(rotation=45, ha="right")  # Rotate labels if they overlap
    plt.tight_layout()


def tabulate_combination_accuracy(data: Dict):
    return pd.DataFrame.from_dict(data, orient="index", columns=["Value", "Count"])


def plot_combination_accuracy(df: pd.DataFrame):
    raise


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
    path = os.path.join("archive", "gtgv0", "results")
    main(path)
