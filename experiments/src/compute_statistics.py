import itertools
import json
import logging
import math
import os
import pickle
import random
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

from experiments.src.file_handler import write_results
from experiments.src.utils import (
    convert_numpy_to_float,
)


def calculate_statistics(
    concepts: list[str],
    activations: dict[str, Any],
    labels: dict[str, Any],
    probes: dict[str, LogisticRegression],
    result_path: str = "experiments/results",
    filename: str = "probe_statistics.json",
):
    stats = {}
    for concept in concepts:
        latest_activations = {"latest": list(activations[concept])}
        c_labels = list(labels[concept]).copy()
        stat = calculate_statistic(
            latest_activations,
            c_labels,
            {"latest": probes[concept]},
            probes,
        )
        stats[concept] = stat

    path = os.path.join(result_path, filename)
    stats = convert_numpy_to_float(stats)
    write_results(stats, path)
    return stats


def calculate_statistic(
    activations: dict[str, dict[str, np.ndarray]],
    labels: Any,
    probe: dict[str, dict[str, LogisticRegression]],
    probes: dict[str, dict[str, LogisticRegression]],
):
    """
    Compute statistical and geometric metrics from neural activations and probe coefficients.

    Args:
        activations (dict[str, dict[str, np.ndarray]]):
            Dictionary of activation arrays, keyed by layer names. Expects "latest" key.
        probes (dict[str, dict[str, LogisticRegression]]):
            Dictionary of fitted LogisticRegression probes, keyed by layer names. Expects "latest" key.

    Returns:
        dict: Dictionary containing:
            - "mean": Mean of flattened activations.
            - "variance": Variance of flattened activations.
            - "std_dev": Standard deviation of flattened activations.
            - "median": Median of flattened activations.
            - "range": Range (max-min) of flattened activations.
            - "iq_range": Interquartile range of flattened activations.
            - "density": Inverse mean pairwise Mahalanobis (or Euclidean) distance between activations.
            - "accuracy": Cosine similarity between centroid and probe CAV.
            - "distance": Norm of vector perpendicular to centroid projection onto CAV.
            - "coherence":
            - "seperation":
            - "information gain":

    Handles exceptions in distance calculation, falling back to Euclidean if Mahalanobis fails.
    """
    layer_activations = activations["latest"]
    latest_probe = probe["latest"]
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
    cav = np.array(latest_probe.coef_)
    cavs = np.array([p.coef_ for p in probes.values()])
    accuracy = cosine_similarity(centroid.reshape(-1, 1), cav.reshape(-1, 1))
    centroid = centroid.flatten()
    cav = cav.flatten()

    projection = (np.dot(centroid, cav) / np.dot(cav, cav)) * cav
    vector_perp = cav - projection
    distance = np.linalg.norm(vector_perp)

    similarity_matrix = torch.matmul(torch.tensor(points), torch.tensor(cav.T))
    K = 200
    _, top_k_indices = torch.topk(similarity_matrix, K)

    coherence_term = 0
    for m in range(points.shape[0]):
        coherence_term += torch.sum(
            torch.matmul(torch.tensor(cav), torch.tensor(points).T)
        )
    coherence_term /= points.shape[0]
    coherence_term = float(coherence_term)

    cav_similarities = torch.matmul(torch.tensor(cav), torch.tensor(cavs).T)
    mask = 1.0 - torch.eye(cavs.shape[0], device=cavs.device)
    separation_term = float(torch.sum(torch.abs(cav_similarities * mask)))

    labels = torch.tensor(labels)
    unique_labels, counts = torch.unique(labels, return_counts=True)
    max_count = max(counts)
    information_gain_term = float(max_count / labels.numel())

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
        "coherence": coherence_term,
        "separation": separation_term,
        "information_gain": information_gain_term,
    }
    print(stats)
    return stats


def collect_and_compute_variance(
    base_filenames,  # e.g. ['feature_importances', 'randomforest_feature_importances', ...]
    result_dir="experiments/results",
    num_iterations=10,
    output_file="importance_variance_stats.pkl",
):
    output_file = os.path.join(result_dir, output_file)
    method_count = len(base_filenames)
    stats = []
    for idx in range(num_iterations):
        files = [
            os.path.join(result_dir, f"{base}_{idx}.json") for base in base_filenames
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
                importances = (importances - importances.min()) / (
                    importances.max() - importances.min()
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


def collect_and_compute_agreement(
    base_filenames,
    result_dir="experiments/results",
    num_iterations=10,
    output_file="importance_agreement_stats.pkl",
    method: Literal["rbo", "topk"] = "rbo",  # or "topk"
    k=10,  # only used for topk or for rbo top-k
    rbo_p=0.9,
):
    output_file = os.path.join(result_dir, method + "_" + output_file)
    method_count = len(base_filenames)
    stats = []
    for idx in range(num_iterations):
        files = [
            os.path.join(result_dir, f"{base}_{idx}.json") for base in base_filenames
        ]
        data = []
        for file in files:
            with open(file, "r") as f:
                d = json.load(f)
                importances = np.array(
                    [v[0] for k, v in sorted(d.items(), key=lambda x: int(x[0]))]
                )
                importances = (importances - importances.min()) / (
                    importances.max() - importances.min()
                )
                data.append(importances)
        data = np.array(data)  # shape: (methods, features)
        # For all group sizes (2 ... method_count)
        for group_size in range(2, method_count + 1):
            for group in itertools.combinations(range(method_count), group_size):
                subset = data[list(group), :]  # shape: (group_size, features)
                # Compute agreement between all pairs in the group
                agreements = []
                for i, j in itertools.combinations(range(group_size), 2):
                    if method == "rbo":
                        v = rbo_from_importances(subset[i], subset[j], k=k, p=rbo_p)
                    elif method == "topk":
                        v = topk_jaccard(subset[i], subset[j], k=k)
                    else:
                        raise ValueError("method must be 'rbo' or 'topk'")
                    agreements.append(v)
                mean_agr = np.mean(agreements)
                n = len(agreements)
                if n > 1:
                    se = np.std(agreements, ddof=1) / np.sqrt(n)
                    ci_range = se * t.ppf(0.975, n - 1)
                else:
                    ci_range = 0.0
                stats.append(
                    {
                        "iteration": idx,
                        "group_methods": [base_filenames[i] for i in group],
                        "mean_agreement": mean_agr,
                        "ci_lower": mean_agr - ci_range,
                        "ci_upper": mean_agr + ci_range,
                        "metric": method,
                        "k": k if method == "topk" or k else None,
                    }
                )
    with open(output_file, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved agreement statistics to {output_file}")


def topk_jaccard(a, b, k):
    a_topk = set(np.argsort(a)[-k:][::-1])
    b_topk = set(np.argsort(b)[-k:][::-1])
    intersection = len(a_topk & b_topk)
    union = len(a_topk | b_topk)
    return intersection / union if union > 0 else 0.0


def rbo(list1, list2, p=0.9):
    s, l = (list1, list2) if len(list1) <= len(list2) else (list2, list1)
    s_set, l_set = set(), set()
    rbo_score = 0.0
    for d in range(len(l)):
        if d < len(s):
            s_set.add(s[d])
        l_set.add(l[d])
        overlap = len(s_set & l_set)
        rbo_score += overlap / (d + 1) * pow(p, d)
    rbo_score *= 1 - p
    return rbo_score


def rbo_from_importances(a, b, k=None, p=0.9):
    a_order = np.argsort(a)[::-1]
    b_order = np.argsort(b)[::-1]
    if k:
        a_order = a_order[:k]
        b_order = b_order[:k]
    return rbo(list(a_order), list(b_order), p=p)


def collect_accuracies(
    results_dir="experiments/results/accuracies",
    output_dir="experiments/results/",
    output_file="accuracies.json",
):
    output_file = os.path.join(output_dir, output_file)
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
    results_dir="experiments/results",
    num_iterations=10,
    output_file="max_accuracies.pkl",
):
    output_file = os.path.join(results_dir, output_file)
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


def calculate_shapley_values(
    results, concepts: list[str], num_permutations: int = 1000
):
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

    shapley_values = defaultdict(list)
    for _ in range(num_permutations):
        perm = list(concepts)
        random.shuffle(perm)
        coalition = ()
        for concept in perm:
            prev = tuple(sorted(coalition))
            coalition = tuple(sorted(list(coalition) + [concept]))
            if prev not in results or coalition not in results:
                continue
            marginal = results[prev][1] - results[coalition][1]
            shapley_values[concept].append(marginal)

    shapley_values = {
        c: (sum(vs) / len(vs) if vs else 0.0) for c, vs in shapley_values.items()
    }
    return shapley_values


def obs_diff(obs: torch.Tensor, other_obs: list[torch.Tensor]) -> float:
    diffs = [pearson_correlation(obs, o) for o in other_obs]
    # diffs = [compare_matrices_abs(obs, o) for o in other_obs]
    obs_diff = sum(diffs)
    max_diff = len(other_obs)
    diff = obs_diff / max_diff
    return float(diff)
