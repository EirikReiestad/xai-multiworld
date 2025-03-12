import gc
import logging
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.fixes import pd
from tabulate import tabulate
from torch.utils.data import TensorDataset

from multiworld.base import MultiWorldEnv
from utils.common.code import get_memory_usage
from utils.common.collections import get_combinations
from utils.common.model_artifact import ModelArtifact
from utils.common.numpy_collections import convert_numpy_to_float
from utils.common.observation import (
    Observation,
    zip_observation_data,
)
from utils.common.read import read_results
from utils.common.write import write_results
from utils.core.plotting import plot_3d
from xailib.common.activations import compute_activations_from_models
from xailib.common.concept_score import (
    binary_concept_scores,
    individual_binary_concept_score,
    individual_soft_concept_score,
)
from xailib.common.probes import get_probe
from xailib.common.tcav_score import tcav_scores
from xailib.common.train_model import train_decision_tree, train_model
from xailib.core.network.feed_forward import FeedForwardNetwork
from xailib.utils.logging import log_shapley_values, log_similarity, log_stats


def get_tcav_scores(
    concepts: List[str],
    test_activations: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    test_output: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    probes: Dict[str, Dict[str, Dict[str, LogisticRegression]]],
    results_path: str = os.path.join("assets", "results"),
    figure_path: str = os.path.join("assets", "figures"),
    show: bool = False,
) -> Dict[str, Dict[str, float]]:
    concept_tcav_scores = {}

    for concept in concepts:
        scores = tcav_scores(
            test_activations[concept], test_output[concept], probes[concept]
        )
        concept_tcav_scores[concept] = scores

        plot_3d(
            scores,
            label=concept,
            folder_path=figure_path,
            filename="tcav_" + concept,
            min=0,
            max=1,
            show=show,
        )
    write_results(
        concept_tcav_scores,
        os.path.join(results_path, "tcav_scores.json"),
    )
    return concept_tcav_scores


def get_concept_scores(
    concepts: List[str],
    test_activations: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    probes: Dict[str, Dict[str, Dict[str, LogisticRegression]]],
    results_path: str = os.path.join("assets", "results"),
    figure_path: str = os.path.join("assets", "figures"),
    show: bool = False,
) -> Dict[str, Dict[str, float]]:
    concept_scores = {}

    for concept in concepts:
        concept_score = binary_concept_scores(
            test_activations[concept], probes[concept]
        )
        concept_scores[concept] = concept_score

        plot_3d(
            concept_score,
            label=concept,
            folder_path=figure_path,
            filename=f"concept_score_{concept}",
            min=0,
            max=1,
            show=show,
        )
    write_results(concept_scores, os.path.join(results_path, "concept_scores.json"))
    return concept_scores


def compute_accuracy_decision_tree(
    concepts: List[str],
    activations: Dict[str, Dict],
    labels: List[int],
    probes: Dict[str, LogisticRegression],
    layer_idx: int,
    concept_score_method: Literal["binary", "soft"],
    epochs: int,
    result_path: str,
    figure_path: str,
    filename: str,
    verbose: bool = False,
):
    test_split = 0.2

    concept_scores = np.array(
        get_concept_score(
            activations, probes, layer_idx, concept_score_method=concept_score_method
        ),
        dtype=np.float32,
    )
    dataset = TensorDataset(
        torch.from_numpy(concept_scores), torch.tensor(labels, dtype=torch.long)
    )
    model = DecisionTreeClassifier()

    train_decision_tree(
        model=model,
        dataset=dataset,
        test_split=test_split,
        feature_names=concepts,
        epochs=epochs,
        result_path=result_path,
        figure_path=figure_path,
        verbose=verbose,
    )


def compute_accuracy(
    observation_shape: int,
    action_shape: int,
    activations: Dict[str, Dict],
    labels: List[int],
    probes: Dict[str, LogisticRegression],
    concept_score_method: Literal["binary", "soft"],
    layer_idx: int,
    hidden_units: int = 500,
    epochs: int = 500,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    test_split: float = 0.1,
    verbose: bool = False,
):
    concept_scores = np.array(
        get_concept_score(
            activations, probes, layer_idx, concept_score_method=concept_score_method
        ),
        dtype=np.float32,
    )

    model = FeedForwardNetwork(observation_shape, hidden_units, action_shape)
    dataset = TensorDataset(
        torch.from_numpy(concept_scores), torch.tensor(labels, dtype=torch.long)
    )
    loss, accuracy = train_model(
        model=model,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        dataset=dataset,
        test_split=test_split,
        val_split=val_split,
        verbose=verbose,
    )

    del model, dataset
    gc.collect()

    return loss, accuracy


def get_concept_score(
    activations: Dict[str, Dict],
    probes: Dict[str, LogisticRegression],
    layer_idx: int,
    concept_score_method: Literal["binary", "soft"] = "binary",
):
    layer_activations = list(activations["latest"].values())[layer_idx]

    concept_scores = []
    for concept, probe in probes.items():
        if concept_score_method == "binary":
            concept_score = individual_binary_concept_score(layer_activations, probe)
        elif concept_score_method == "soft":
            concept_score = individual_soft_concept_score(layer_activations, probe)
        else:
            raise ValueError(
                f"Concept score method {concept_score_method} not recognized."
            )
        concept_scores.append(concept_score)

    concept_scores = list(zip(*concept_scores))

    return concept_scores


def calculate_shapley_values(path: str, concepts: List[str]):
    results = read_results(path)

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
    N = len(concepts)
    for concept in concepts:
        other_concepts = concepts.copy()
        other_concepts.remove(concept)
        for comb in get_combinations(other_concepts):
            comb_u_concept = tuple(sorted(comb + [concept]))
            comb = tuple(sorted(comb))
            coalition_size = len(comb)
            factorial_term = (
                math.factorial(coalition_size)
                * math.factorial(N - coalition_size - 1)
                / math.factorial(N)
            )

            accuracy = results[comb_u_concept][1]
            comb_accuracy = results[comb][1]

            marginal_contribution = accuracy - comb_accuracy
            shapley_values[concept] += factorial_term * marginal_contribution
    log_shapley_values(shapley_values)
    return shapley_values


def calculate_probe_robustness(
    concepts: List[str],
    model: nn.Module,
    splits: List[float],
    layer_idx: int,
    epochs: int,
    results_path: str = os.path.join("assets", "results"),
    ignore_layers: List[str] = [],
):
    concept_similarities = defaultdict(lambda: defaultdict(float))
    for _ in range(epochs):
        concept_probe_robustness = probe_robustness(
            concepts, layer_idx, model, splits, ignore_layers=ignore_layers
        )
        for concept in concepts:
            for key, value in concept_probe_robustness[concept].items():
                concept_similarities[concept][key] += value

    for concept in concepts:
        for key in concept_similarities[concept].keys():
            concept_similarities[concept][key] /= epochs

    path = os.path.join(results_path, "probe_robustness.json")
    concept_similarities = convert_numpy_to_float(concept_similarities)
    write_results(concept_similarities, path)
    log_similarity(concept_similarities, concepts)

    return concept_similarities


def probe_robustness(
    concepts: List[str],
    layer_idx: int,
    model: nn.Module,
    splits: List[float],
    ignore_layers: List[str] = [],
) -> Dict:
    concept_similarities = {}
    for concept in concepts:
        similarities = get_probe_similarity(
            concept, layer_idx, model, splits, ignore_layers=ignore_layers
        )
        concept_similarities[concept] = similarities

    return concept_similarities


def get_probe_similarity(
    concept: str,
    layer_idx: int,
    model: nn.Module,
    splits: List[float],
    ignore_layers: List[str] = [],
) -> Dict[float, float]:
    base = get_probe(concept, layer_idx, model, 1.0, ignore_layers=ignore_layers)
    if base is None:
        return {}

    probe_splits = {}
    for split in splits:
        probe = get_probe(concept, layer_idx, model, split, ignore_layers=ignore_layers)
        if probe is None:
            continue
        probe_splits[split] = probe

    similarities = {
        key: cosine_similarity(base.coef_, probe.coef_)
        for key, probe in probe_splits.items()
    }
    return similarities


def calculate_statistics(
    concepts: List[str],
    activations: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    probes: Dict[str, Dict[str, Dict[str, LogisticRegression]]],
    layer_idx: int,
    results_path: str = os.path.join("assets", "results"),
    filename: str = "probe_statistics.json",
):
    stats = {}
    for concept in concepts:
        latest_activations = {"latest": list(activations[concept].values())[-1]}
        stat = calculate_statistic(
            latest_activations,
            {"latest": list(probes[concept].values())[-1]},
            layer_idx,
        )
        stats[concept] = stat

    path = os.path.join(results_path, filename)
    stats = convert_numpy_to_float(stats)
    write_results(stats, path)
    log_stats(stats)


def calculate_statistic(
    activations: Dict[str, Dict[str, np.ndarray]],
    probes: Dict[str, Dict[str, LogisticRegression]],
    layer_idx: int,
):
    layer_activations = list(activations["latest"].values())[layer_idx]["output"]
    probe = list(probes["latest"].values())[layer_idx]
    points = layer_activations.detach().numpy()

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
    cav = probe.coef_
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


def calculate_probe_similarities(
    probes: Dict[
        str, Dict[str, Dict[str, LogisticRegression]]
    ],  # Concept -> Model -> Layer -> Probe
    layer_idx: int,
    result_path: str = os.path.join("assets", "results"),
    filename: str = "cos_sim_matrix.json",
):
    cavs = {}
    for key, value in probes.items():
        probe = list(list(value.values())[-1].values())[layer_idx]
        cavs[key] = probe.coef_.flatten()
    calculate_cav_similarity(cavs, result_path, filename=filename)


def calculate_cav_similarity(
    cavs: Dict[str, np.ndarray], result_path: str, filename: str = "cos_sim_matrix.json"
):
    coefs = np.array(list(cavs.values()))

    cos_sim_matrix = cosine_similarity(coefs)
    cos_sim_matrix_df = pd.DataFrame(
        cos_sim_matrix, index=cavs.keys(), columns=cavs.keys()
    )
    output_file = os.path.join(result_path, filename)
    cos_sim_matrix_df.to_json(output_file, orient="index")

    logging.info(f"\n{cos_sim_matrix_df}")
