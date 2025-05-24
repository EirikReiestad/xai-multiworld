import itertools
import json
import logging
import math
import os
import pickle
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from experiments.src.file_handler import read_results, write_results
from experiments.src.utils import (
    convert_numpy_to_float,
    get_combinations,
    log_shapley_values,
)
from tabulate import tabulate


def calculate_statistics(
    concepts: list[str],
    activations: dict[str, Any],
    probes: dict[str, LogisticRegression],
    results_path: str = "experiments/results",
    filename: str = "probe_statistics.json",
):
    stats = {}
    for concept in concepts:
        latest_activations = {"latest": list(activations[concept])}
        stat = calculate_statistic(
            latest_activations,
            {"latest": probes[concept]},
        )
        stats[concept] = stat

    path = os.path.join(results_path, filename)
    stats = convert_numpy_to_float(stats)
    write_results(stats, path)
    return stats


def calculate_statistic(
    activations: dict[str, dict[str, np.ndarray]],
    probes: dict[str, dict[str, LogisticRegression]],
):
    layer_activations = activations["latest"]
    probe = probes["latest"]
    points = np.array(layer_activations)

    n_samples = points.shape[0]
    points = points.reshape(n_samples, -1)

    mean = np.mean(points)
    variance = np.var(points)
    std_dev = np.std(points)
    median = np.median(points)
    range_val = np.ptp(points)

    q75, q25 = np.percentile(points, [75, 25])
    iq_range = q75 - q25

    try:
        cov_matrix = np.cov(points, rowvar=False)
        reg_term = 1e-5 * np.eye(cov_matrix.shape[0])
        inv_cov_matrix = np.linalg.inv(cov_matrix + reg_term)
        pairwise_distances = pdist(points, "mahalanobis", VI=inv_cov_matrix)
    except ValueError as e:
        logging.warning(e)
        logging.info("Running pdist with euclidean distance")
        pairwise_distances = pdist(points, "euclidean")
    mean_distance = np.mean(pairwise_distances)
    density = 1 / mean_distance if mean_distance != 0 else float("inf")
    centroid = np.mean(points, axis=0)
    cav = np.array(probe.coef_)
    accuracy = cosine_similarity(centroid.reshape(-1, 1), cav.reshape(-1, 1))
    centroid = centroid.flatten()
    cav = cav.flatten()

    projection = (np.dot(centroid, cav) / np.dot(cav, cav)) * cav
    vector_perp = cav - projection
    distance = np.linalg.norm(vector_perp)

    stats = {
        "mean": float(mean),
        "variance": float(variance),
        "std_dev": float(std_dev),
        "median": float(median),
        "range": float(range_val),
        "iq_range": float(iq_range),
        "density": density,
        "accuracy": accuracy[0][0],
        "distance": distance,
    }
    return stats


def collect_and_compute_variance(
    base_filenames,  # e.g. ['feature_importances', 'randomforest_feature_importances', ...]
    results_dir="experiments/results",
    num_iterations=10,
    output_file="experiments/results/importance_variance_stats.pkl",
):
    method_count = len(base_filenames)
    stats = []
    for idx in range(num_iterations):
        files = [
            os.path.join(results_dir, f"{base}_{idx}.json") for base in base_filenames
        ]
        # Load importances for each method
        data = []
        for file in files:
            with open(file, "r") as f:
                # Each value is (importance, split_count), use importance
                d = json.load(f)
                # Sorted by feature index
                importances = np.array(
                    [v[0] for k, v in sorted(d.items(), key=lambda x: int(x[0]))]
                )
                importances = (
                    importances / importances.sum()
                    if importances.sum() > 0
                    else importances
                )
                data.append(importances)
        data = np.array(data)  # shape: (methods, features)
        M = data.shape[1]
        # For all group sizes (2 ... method_count)
        for group_size in range(2, method_count + 1):
            for group in itertools.combinations(range(method_count), group_size):
                subset = data[list(group), :]  # shape: (group_size, M)
                variances = np.var(subset, axis=0, ddof=1)  # variance for each feature
                mean_var = np.mean(variances)
                # 95% CI for the mean variance, using t-distribution
                n = len(variances)
                if n > 1:
                    se = np.std(variances, ddof=1) / np.sqrt(n)
                    ci_range = se * t.ppf(0.975, n - 1)
                else:
                    ci_range = 0.0
                stats.append(
                    {
                        "iteration": idx,
                        "group_methods": [base_filenames[i] for i in group],
                        "mean_variance": mean_var,
                        "ci_lower": mean_var - ci_range,
                        "ci_upper": mean_var + ci_range,
                    }
                )
    with open(output_file, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved variance statistics to {output_file}")


def collect_accuracies(
    results_dir="experiments/results/accuracies",
    output_file="experiments/results/accuracies.json",
):
    dir_list = os.listdir(results_dir)

    results = defaultdict(lambda: defaultdict(list))

    for file in dir_list:
        if not file.endswith(".json"):
            continue
        lambda_1, lambda_2, lambda_3, iteration = file.removesuffix(".json").split("_")
        lambda_1 = round(float(lambda_1), 2)
        lambda_2 = round(float(lambda_2), 2)
        lambda_3 = round(float(lambda_3), 2)
        main_key = f"{lambda_1};{lambda_2};{lambda_3}"
        file_path = os.path.join(results_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                results[main_key][key].append(value)

    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Saved accuracies to {output_file}")


def collect_max_accuracy(
    base_filenames: list[str],
    other_files: list[str] = [],
    results_dir="experiments/results",
    num_iterations=10,
    output_file="experiments/results/max_accuracies.pkl",
):
    max_accuracies = [[] for _ in range(len(base_filenames))]
    for idx in range(num_iterations):
        files = [
            os.path.join(results_dir, f"{base}_{idx}.json") for base in base_filenames
        ]
        # Load importances for each method
        for i, file in enumerate(files):
            with open(file, "r") as f:
                # Each value is (importance, split_count), use importance
                d = json.load(f)
                # Sorted by feature index
                importances = np.array(
                    [v[0] for k, v in sorted(d.items(), key=lambda x: int(x[0]))]
                )
                importances = (
                    importances / importances.sum()
                    if importances.sum() > 0
                    else importances
                )
                max_accuracies[i].append(max(importances))

        files = [os.path.join(f"{base}_{idx}.json") for base in base_filenames]

    with open(output_file, "wb") as f:
        pickle.dump(max_accuracies, f)
    print(f"Saved max accuracies to {output_file}")


def calc_similarity_matrix(average_positive_observations, positive_observations):
    diff_matrix = defaultdict(dict)
    for key, value in positive_observations.items():
        for other_key, other_value in positive_observations.items():
            diff_matrix[key][other_key] = obs_diff(
                average_positive_observations[key], other_value
            )
    df = pd.DataFrame(diff_matrix)
    print(f"Similarity matrix:\n{df.head()}")
    # Convert to serializable format:
    similarity_matrix = {}
    for key in diff_matrix:
        similarity_matrix[key] = {k: float(v) for k, v in diff_matrix[key].items()}
    return similarity_matrix


def pearson_correlation(A, B):
    A_flat = A.flatten()
    B_flat = B.flatten()
    correlation = np.corrcoef(A_flat, B_flat)[0, 1]
    return correlation


def calculate_shapley_values(results, concepts: list[str]):
    table_accuracy = sorted(
        [(comb, loss, accuracy) for comb, (loss, accuracy) in results.items()],
        key=lambda x: x[2],
        reverse=True,
    )
    table_loss = sorted(
        [(comb, loss, accuracy) for comb, (loss, accuracy) in results.items()],
        key=lambda x: x[1],
    )
    logging.info(
        "\n" + tabulate(table_accuracy, headers=["Combination", "Loss", "Accuracy"])
    )
    logging.info(
        "\n" + tabulate(table_loss, headers=["Combination", "Loss", "Accuracy"])
    )

    shapley_values = defaultdict(float)
    M = len(concepts)
    for concept in concepts:
        other_concepts = concepts.copy()
        other_concepts.remove(concept)
        for comb in get_combinations(other_concepts):
            comb_u_concept = tuple(sorted(comb + [concept]))
            if comb_u_concept not in results or tuple(comb) not in results:
                continue
            print(comb, comb_u_concept)
            comb = tuple(sorted(comb))
            coalition_size = len(comb)
            factorial_term = (
                math.factorial(coalition_size)
                * math.factorial(M - coalition_size - 1)
                / math.factorial(M)
            )

            accuracy = results[comb_u_concept][0]
            comb_accuracy = results[comb][0]

            marginal_contribution = accuracy - comb_accuracy
            shapley_values[concept] += factorial_term * marginal_contribution
    log_shapley_values(shapley_values)
    return shapley_values


def obs_diff(obs: torch.Tensor, other_obs: list[torch.Tensor]) -> float:
    diffs = [pearson_correlation(obs, o) for o in other_obs]
    # diffs = [compare_matrices_abs(obs, o) for o in other_obs]
    obs_diff = sum(diffs)
    max_diff = len(other_obs)
    diff = obs_diff / max_diff
    return float(diff)
