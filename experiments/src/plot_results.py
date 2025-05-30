import json
import os
import pickle
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from experiments.src.constants import get_colormap, get_palette


def ensure_plots_dir(directory="experiments/plots"):
    os.makedirs(directory, exist_ok=True)


def set_plot_styles():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )


def load_results(filename):
    with open(filename, "r") as f:
        return json.load(f)


def df_from_results(results):
    return pd.DataFrame(
        [{k: v for k, v in r.items() if k != "similarity_matrix"} for r in results]
    )


def filter_cav_and_baseline(df_results):
    df_cav = df_results[df_results["M"] != "baseline"].copy()
    df_cav["M"] = pd.to_numeric(df_cav["M"])
    df_baseline = df_results[df_results["M"] == "baseline"].copy()
    return df_cav, df_baseline


def plot_accuracy_vs_M(df_cav, save_path):
    max_depth = sorted(df_cav["max_depth"].unique())
    depth_palette = get_palette([str(d) for d in max_depth])
    plt.figure(figsize=(12, 8))
    for depth in max_depth:
        subset = df_cav[df_cav["max_depth"] == depth]
        plt.plot(
            subset["M"],
            subset["accuracy"],
            marker="o",
            markersize=10,
            linewidth=2.5,
            color=depth_palette[str(depth)],
            label=f"max_depth={depth}",
        )
    plt.xlabel("M (Number of CAVs)", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title(
        "Accuracy vs Number of CAVs for Different Decision Tree Depths", fontsize=18
    )
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_accuracy_vs_depth(df_cav, df_baseline, save_path):
    m_values = sorted(df_cav["M"].unique())
    m_palette = get_palette([str(m) for m in m_values])
    plt.figure(figsize=(12, 8))
    for m in m_values:
        subset = df_cav[df_cav["M"] == m]
        plt.plot(
            subset["max_depth"],
            subset["accuracy"],
            marker="o",
            markersize=10,
            linewidth=2.5,
            color=m_palette[str(m)],
            label=f"M={m}",
        )
    plt.plot(
        df_baseline["max_depth"],
        df_baseline["accuracy"],
        marker="s",
        markersize=10,
        linestyle="--",
        color="black",
        linewidth=3,
        label="Baseline (raw features)",
    )
    plt.xticks(sorted(df_baseline["max_depth"].unique()), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Max Depth", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title(
        "Accuracy vs Decision Tree Depth for Different Numbers of CAVs", fontsize=18
    )
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_similarity_matrices(results, save_dir):
    custom_cmap = get_colormap()
    for r in results:
        if r["M"] != "baseline" and r["max_depth"] == 2:
            m = r["M"]
            sim_matrix = r["similarity_matrix"]
            matrix_data = {key: value for key, value in sim_matrix.items()}
            df_sim = pd.DataFrame(matrix_data)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                df_sim,
                annot=True,
                annot_kws={"size": 14},
                cmap=custom_cmap,
                fmt=".2f",
                cbar_kws={"label": "Similarity", "shrink": 0.8},
            )
            plt.title(f"Concept Similarity Matrix (M={m})", fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"similarity_matrix_M{m}.png"), dpi=300)
            plt.close()


def summarize_results(df_cav, df_baseline, save_path):
    best_by_m = df_cav.groupby("M")["accuracy"].max().reset_index()
    best_by_m.columns = ["M", "Best Accuracy"]

    best_configs = []
    for _, row in best_by_m.iterrows():
        best_row = df_cav[
            (df_cav["M"] == row["M"]) & (df_cav["accuracy"] == row["Best Accuracy"])
        ].iloc[0]
        best_configs.append(
            {
                "M": int(best_row["M"]),
                "Best max_depth": best_row["max_depth"],
                "Accuracy": best_row["accuracy"],
            }
        )

    df_best = pd.DataFrame(best_configs)
    best_overall_idx = df_best["Accuracy"].idxmax()
    best_overall = df_best.iloc[best_overall_idx]
    best_baseline = df_baseline.loc[df_baseline["accuracy"].idxmax()]

    summary = pd.DataFrame(best_configs)
    summary = summary.sort_values("M")

    summary_text = (
        f"Summary of Results:\n\n"
        f"{summary.to_string(index=False)}\n\n"
        f"Overall Best Configuration:\n"
        f"M={best_overall['M']}, max_depth={best_overall['Best max_depth']}, Accuracy={best_overall['Accuracy']:.4f}\n\n"
        f"Best Baseline (perfect info):\n"
        f"max_depth={best_baseline['max_depth']}, Accuracy={best_baseline['accuracy']:.4f}"
    )

    with open(save_path, "w") as f:
        f.write(summary_text)


def plot_compare_json_files(json_files, labels=None, save_path=None, plot_splits=False):
    """
    Plots importance (and optionally splits) from multiple JSON files.
    X-axis: index (1, 2, 3, ...)
    Y-axis: importance (main), splits (secondary if plot_splits=True)
    """

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = None

    for idx, file in enumerate(json_files):
        with open(file, "r") as f:
            data = json.load(f)
        items = sorted(data.items(), key=lambda x: int(x[0]))
        x = list(range(1, len(items) + 1))
        importance = [v[0] for _, v in items]
        splits = [v[1] for _, v in items]
        # Normalize importance to range 0-1
        if max(importance) != min(importance):
            scaler = MinMaxScaler()
            importance = scaler.fit_transform(
                np.array(importance).reshape(-1, 1)
            ).flatten()
        else:
            importance = [0.0 for _ in importance]
        label = labels[idx] if labels else f"File {idx+1}"
        ax1.plot(x, importance, marker="o", label=f"{label} (importance)")
        if plot_splits:
            if ax2 is None:
                ax2 = ax1.twinx()
            ax2.plot(
                x,
                splits,
                marker="s",
                linestyle="--",
                alpha=0.5,
                label=f"{label} (splits)",
                color=f"C{idx+1}",
            )

    ax1.set_xlabel("Index", fontsize=14)
    ax1.set_ylabel("Importance", fontsize=14)
    ax1.set_title(
        "Comparison of Importance (and Splits)"
        if plot_splits
        else "Comparison of Importance",
        fontsize=16,
    )
    ax1.legend(fontsize=12, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.7)
    if plot_splits and ax2:
        ax2.set_ylabel("Splits", fontsize=14)
        ax2.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def create_visualizations(
    plots_dir="experiments/plots",
    compare_json_files=None,
    compare_labels=None,
):
    """
    ensure_plots_dir(plots_dir)
    set_plot_styles()
    results = load_results(results_file)
    df_results = df_from_results(results)
    df_cav, df_baseline = filter_cav_and_baseline(df_results)

    plot_accuracy_vs_M(df_cav, os.path.join(plots_dir, "accuracy_vs_M.png"))
    plot_accuracy_vs_depth(
        df_cav, df_baseline, os.path.join(plots_dir, "accuracy_vs_depth.png")
    )
    plot_similarity_matrices(results, plots_dir)
    summarize_results(
        df_cav, df_baseline, os.path.join(plots_dir, "summary_results.txt")
    )

    """
    # Plot comparison if files are given
    if compare_json_files:
        plot_compare_json_files(
            compare_json_files,
            labels=compare_labels,
            save_path=os.path.join(plots_dir, "json_file_comparison.png"),
        )

    print(f"Plots and summary generated successfully in the '{plots_dir}' directory!")


def plot_variance_stats(
    result_dir: str = "experiments/results/",
    filename: str = "importance_variance_stats.pkl",
    save_filename: str = "importance_variance_plot.png",
    drop: list[str] = [],
):
    load_path = os.path.join(result_dir, filename)
    save_path = os.path.join(result_dir, save_filename)
    with open(load_path, "rb") as f:
        stats = pickle.load(f)
    grouped = defaultdict(list)
    for s in stats:
        key = tuple(sorted(s["group_methods"]))
        grouped[key].append(s)
    labels = []
    group_labels = []
    mean_variances = []
    means = []
    lowers = []
    uppers = []
    spreads = []
    for group, values in grouped.items():
        if len(group) > 2:
            continue
        label_converter = {
            "feature_importances": "decision tree",
            "feature_importances_15": "decision tree",
            "randomforest_feature_importances": "random forest",
            "xgboost_feature_importances": "xgboost",
            "elasticnet_feature_importances": "elasticnet",
            "logistic_regression_feature_importances": "logistic regression",
            "svm_linear_feature_importances": "svm",
            "nn_feature_importances": "neural network",
        }
        label_group = [label_converter[g] for g in group]
        if any([d.lower() in label_group for d in drop]):
            continue
        mean_variance = [v["mean_variance"] for v in values]
        mean = np.mean([v["mean_variance"] for v in values])
        lower = np.mean([v["ci_lower"] for v in values])
        upper = np.mean([v["ci_upper"] for v in values])
        spread = [upper - lower for lower, upper in zip(lowers, uppers)]
        labels.append(" + ".join(label_group))
        group_labels.extend([" + ".join(label_group)] for _ in values)
        mean_variances.extend(mean_variance)
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)
        spreads.append(spread)
    data = np.array(mean_variances).flatten()
    group_labels = np.array(group_labels).flatten()

    palette = get_palette(labels)
    colors = [palette[i] for i in labels]

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(x=group_labels, y=data, ax=ax, hue=group_labels, palette=palette)
    """
    sns.boxplot(
        data=data,
        ax=ax,
        palette=palette,
        width=0.6,
        dodge=True,
        legend=False,
    )
    """
    # ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(range(1, len(labels) + 1), fontsize=12)

    ax.set_ylabel("Mean Variance of Feature Importances")
    ax.set_title("Uncertainty (Variance) across Method Combinations")
    # Legend outside the plot
    legend_handles = [
        mpatches.Patch(color=colors[i], label=f"{i+1}: {labels[i]}")
        for i in range(len(labels))
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Method Group",
    )
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def plot_agreement_stats(
    result_dir: str = "experiments/results/",
    filename: str = "importance_agreement_stats.pkl",
    save_filename: str = "importance_agreement_plot.png",
    drop: list[str] = [],
):
    load_path = os.path.join(result_dir, filename)
    save_path = os.path.join(result_dir, save_filename)
    with open(load_path, "rb") as f:
        stats = pickle.load(f)
    grouped = defaultdict(list)
    for s in stats:
        key = tuple(sorted(s["group_methods"]))
        grouped[key].append(s)
    labels = []
    group_labels = []
    mean_agreements = []
    means = []
    lowers = []
    uppers = []
    for group, values in grouped.items():
        if len(group) > 2:
            continue
        label_converter = {
            "feature_importances": "decision tree",
            "feature_importances_15": "decision tree",
            "randomforest_feature_importances": "random forest",
            "xgboost_feature_importances": "xgboost",
            "elasticnet_feature_importances": "elasticnet",
            "logistic_regression_feature_importances": "logistic regression",
            "svm_linear_feature_importances": "svm",
            "nn_feature_importances": "neural network",
        }
        label_group = [label_converter.get(g, g) for g in group]
        if any([d.lower() in label_group for d in drop]):
            continue
        mean_agreement = [v["mean_agreement"] for v in values]
        mean = np.mean(mean_agreement)
        lower = np.mean([v["ci_lower"] for v in values])
        upper = np.mean([v["ci_upper"] for v in values])
        labels.append(" + ".join(label_group))
        group_labels.extend([" + ".join(label_group)] * len(values))
        mean_agreements.extend(mean_agreement)
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)
    data = np.array(mean_agreements).flatten()
    group_labels = np.array(group_labels).flatten()

    palette = get_palette(labels)
    colors = [palette[i] for i in labels]

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(x=group_labels, y=data, ax=ax, hue=group_labels, palette=palette)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(range(1, len(labels) + 1), fontsize=12)
    ax.set_ylabel("Mean Agreement Score")
    ax.set_title("Agreement in Feature Importance Rankings across Method Combinations")
    legend_handles = [
        mpatches.Patch(color=colors[i], label=f"{i+1}: {labels[i]}")
        for i in range(len(labels))
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Method Group",
    )
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def plot_accuracy(
    file="accuracies.json",
    plot_model="decision_tree",
    folder="experiments/results",
    save_path="experiments/plots/",
    labels=None,
):
    path = os.path.join(folder, file)
    with open(path) as f:
        data = json.load(f)

    plot_model_to_title = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "elasticnet": "Elasticnet",
        "logistic_regesssion": "Logistic Regression",
        "svm_linear": "Support Vector Machine",
        "nn": "Neural Network",
    }

    records = {"lambda": [], "model": [], "value": []}
    for lambdas, models in data.items():
        for model, vals in models.items():
            for v in vals:
                records["lambda"].append(lambdas)
                records["model"].append(model)
                records["value"].append(v)
    df = pd.DataFrame(records)
    df = df[df["model"] == plot_model]
    df[["lambda1", "lambda2", "lambda3"]] = (
        df["lambda"].str.split(";", expand=True).astype(float)
    )
    mean_df = (
        df.groupby(["lambda1", "lambda2", "lambda3"])["value"].mean().reset_index()
    )

    os.makedirs(save_path, exist_ok=True)
    slices = [
        ("lambda1", ["lambda2", "lambda3"]),
        ("lambda2", ["lambda1", "lambda3"]),
        ("lambda3", ["lambda1", "lambda2"]),
    ]

    """
    for fixed, (var1, var2) in slices:
        subdf = mean_df[mean_df[fixed] == 0]
        if not subdf.empty:
    """

    # pivot = subdf.pivot(index=var2, columns=var1, values="value")
    pivot = df.pivot_table(index=["lambda1"], columns="lambda2", values="value")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=get_colormap(),
        cbar_kws={"label": "Mean Accuracy"},
    )
    plt.title(f"{plot_model_to_title[plot_model]}: Mean Accuracy of lambda combination")
    plt.xlabel("Lambda 1")
    plt.ylabel("Lambda 2")
    out_path = os.path.join(save_path, f"heatmap_{plot_model}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_accuracy_all(
    file="accuracies.json",
    folder="experiments/results",
    save_path="experiments/plots/accuracies.png",
    labels=None,
    avg: bool = False,
):
    path = os.path.join(folder, file)
    with open(path) as f:
        data = json.load(f)

    label_converter = {
        "decision_tree": "decision tree",
        "random_forest": "random forest",
        "xgboost": "xgboost",
        "elasticnet": "elasticnet",
        "logistic_regesssion": "logistic regression",
        "svm_linear": "svm",
        "nn": "neural network",
    }

    records = {
        "lambda": [],
        "model": [],
        "value": [],
    }

    for lambdas, models in data.items():
        for model, vals in models.items():
            for v in vals:
                records["lambda"].append(lambdas)
                records["model"].append(model)
                records["value"].append(v)

    df = pd.DataFrame(records)
    df = df[~df["model"].str.contains("elasticnet", case=False)]

    if avg:
        df = df.drop(columns=["lambda"])
        labels = labels or df["model"].unique().tolist()
        df["model"] = df["model"].map(label_converter).fillna(df["model"])
        color_labels = df["model"].unique().tolist()
        palette = get_palette(list(color_labels))
        colors = [palette[label] for label in color_labels]

        plt.figure(figsize=(14, 7))
        sns.boxplot(
            data=df,
            x="model",
            y="value",
            hue="model",
            palette=palette,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
            },
        )
    else:
        color_labels = df["model"].unique().tolist()
        labels = labels or color_labels
        labels = [label_converter[l] if l in label_converter else l for l in labels]
        palette = get_palette(list(color_labels))
        colors = [palette[label] for label in color_labels]

        plt.figure(figsize=(14, 7))
        sns.pointplot(
            data=df,
            x="model",
            y="value",
            hue="model",
            palette=palette,
            dodge=0.3,
            linestyle="none",
            errorbar="sd",
            capsize=0.1,
            markers="o",
        )
    """
    sns.boxplot(
        data=df,
        x="model",
        y="value",
        hue="lambda",
        palette=palette,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )
    """
    plt.xlabel("Model", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title("CAV Experiment Accuracy", fontsize=18)
    plt.xticks(rotation=20, fontsize=14)
    plt.yticks(fontsize=14)
    legend_handles = [
        mpatches.Patch(color=colors[i], label=f"{i+1}: {color_labels[i]}")
        for i in range(len(color_labels))
    ]
    plt.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Experiment",
        fontsize=14,
        title_fontsize=16,
    )
    if avg:
        plt.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title="Model",
            fontsize=14,
            title_fontsize=16,
        )

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    # plt.show()


def plot_accuracy_ternary(
    file="accuracies.json",
    plot_model="decision_tree",
    folder="experiments/results",
    save_path="experiments/plots",
    labels=None,
):
    path = os.path.join(folder, file)
    with open(path) as f:
        data = json.load(f)

    records = {"lambda": [], "model": [], "value": []}
    for lambdas, models in data.items():
        for model, vals in models.items():
            for v in vals:
                records["lambda"].append(lambdas)
                records["model"].append(model)
                records["value"].append(v)
    df = pd.DataFrame(records)
    df = df[df["model"] == plot_model]
    df[["lambda1", "lambda2", "lambda3"]] = (
        df["lambda"].str.split(";", expand=True).astype(float)
    )
    stats_df = (
        df.groupby(["lambda1", "lambda2", "lambda3"])
        .agg(mean_value=("value", "mean"), variance=("value", "var"))
        .reset_index()
    )
    # plotly requires variance to be positive and not NaN for size
    stats_df["variance"] = stats_df["variance"].fillna(0)
    fig = px.scatter_ternary(
        stats_df,
        a="lambda1",
        b="lambda2",
        c="lambda3",
        color="mean_value",
        size="variance",
        size_max=20,
        color_continuous_scale="Viridis",
        title=f"{plot_model} Mean Accuracy (Variance as Circle Size)",
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
    save_path = os.path.join(save_path, f"accuracies_ternary_{plot_model}.html")
    fig.write_html(save_path)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    # Example usage for the comparison plot:
    base_dirs = [
        "experiments/results_completeness_15",
        # "experiments/results_importance_15",
        # "experiments/results_completeness_lambda_0_1_5_15",
        # "experiments/results_completeness_lambda_0_-1_5_15",
        # "experiments/results_completeness_lambda_-1_1_5_15",
        # "experiments/results_completeness_lambda_-1_1_10_15",
    ]

    compare_json_files = [
        "feature_importances_15_0.json",
        "randomforest_feature_importances_0.json",
        "xgboost_feature_importances_0.json",
        "elasticnet_feature_importances_0.json",
        "logistic_regression_feature_importances_0.json",
        "svm_linear_feature_importances_0.json",
        # "nn_feature_importances_0.json",
    ]
    compare_labels = [
        "decision tree",
        "random forest",
        "xgboost",
        "elasticnet",
        "logistic regression",
        "svm linear",
        "neural network",
    ]

    for base in base_dirs:
        json_files = [os.path.join(base, json_file) for json_file in compare_json_files]
        create_visualizations(
            plots_dir=base,
            compare_json_files=json_files,
            compare_labels=compare_labels,
        )
        plot_variance_stats(
            result_dir=base,
            drop=["elasticnet"],
            filename="importance_variance_stats.pkl",
            save_filename="importance_variance_plot.png",
        )
        plot_agreement_stats(
            result_dir=base,
            filename="rbo_importance_agreement_stats.pkl",
            drop=["elasticnet"],
            save_filename="rbo_importance_variance_plot.png",
        )
        plot_agreement_stats(
            result_dir=base,
            drop=["elasticnet"],
            filename="topk_importance_agreement_stats.pkl",
            save_filename="topk_importance_variance_plot.png",
        )
        for model in [
            "decision_tree",
            "xgboost",
            "random_forest",
            "svm_linear",
            "logistic_regesssion",
            "nn",
        ]:
            plot_accuracy(
                file="accuracies.json",
                plot_model=model,
                folder=base,
                save_path=base,
            )
            continue
            plot_accuracy_ternary(
                file="accuracies.json",
                plot_model=model,
                folder=base,
                save_path=base,
            )
        plot_accuracy_all(
            file="accuracies.json",
            folder=base,
            save_path=os.path.join(base, "accuracies_avg.png"),
            avg=True,
        )
        plot_accuracy_all(
            file="accuracies.json",
            folder=base,
            save_path=os.path.join(base, "accuracies.png"),
        )
