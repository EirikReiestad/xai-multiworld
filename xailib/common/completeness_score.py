import logging
import os
from typing import Dict, List, Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from utils.common.collections import get_combinations
from utils.common.observation import (
    Observation,
    load_and_split_observation,
    zip_observation_data,
)
from utils.common.write import write_results
from xailib.common.activations import compute_activations_from_models
from xailib.common.probes import get_probes
from xailib.utils.metrics import (
    calculate_shapley_values,
    compute_accuracy,
    compute_accuracy_decision_tree,
)


def get_completeness_score(
    probes: Dict[str, Dict[str, Dict[str, LogisticRegression]]],
    concepts: List[str],
    model: nn.Module,
    observations: Observation,
    layer_idx: int,
    epochs: int = 10,
    ignore_layers: List = [],
    method: Literal["decisiontree", "network"] = "network",
    concept_score_method: Literal["soft", "binary"] = "binary",
    verbose: bool = False,
    result_path: str = os.path.join("assets", "results"),
    figure_path: str = os.path.join("assets", "figures"),
    filename: str = "completeness_score.json",
):
    if method == "network":
        return get_completeness_score_network(
            model=model,
            probes=probes,
            observations=observations,
            layer_idx=layer_idx,
            concepts=concepts.copy(),
            concept_score_method=concept_score_method,
            epochs=epochs,
            ignore_layers=ignore_layers,
            verbose=verbose,
            result_path=result_path,
            filename=filename,
        )
    elif method == "decisiontree":
        return get_completeness_score_decision_tree(
            model=model,
            probes=probes,
            observations=observations,
            layer_idx=layer_idx,
            concepts=concepts,
            concept_score_method=concept_score_method,
            epochs=epochs,
            ignore_layers=ignore_layers,
            verbose=verbose,
            result_path=result_path,
            figure_path=figure_path,
            filename=filename,
        )


def get_completeness_score_decision_tree(
    model: nn.Module,
    probes: Dict[str, Dict[str, Dict[str, LogisticRegression]]],
    observations: Observation,
    layer_idx: int,
    concepts: List[str],
    concept_score_method: Literal["soft", "binary"],
    epochs: int = 10,
    ignore_layers: List[str] = [],
    result_path: str = os.path.join("assets", "results"),
    figure_path: str = os.path.join("assets", "figures"),
    verbose: bool = False,
    filename: str = "completeness_score_decision_tree.json",
):
    models = {"latest": model}

    observation_zipped, _ = zip_observation_data(observations)
    activations, input, output = compute_activations_from_models(
        models, observation_zipped, ignore_layers
    )
    labels = np.array(observations[..., Observation.LABEL], dtype=np.float32)
    # labels = torch.argmax(output["latest"], dim=1).detach().numpy()
    # np.random.shuffle( labels)  # This is just for sanity check, where it should score lower
    action_space = len(output["latest"][0])
    unique_elements, counts = np.unique(labels, return_counts=True)
    for element, count in zip(unique_elements, counts):
        logging.info(f"Element {element} occurs {count} times.")

    concept_probes = {
        key: list(list(value.values())[-1].values())[layer_idx]
        for key, value in probes.items()
    }

    return compute_accuracy_decision_tree(
        concepts=concepts,
        activations=activations,
        labels=labels,
        probes=concept_probes,
        layer_idx=layer_idx,
        concept_score_method=concept_score_method,
        epochs=epochs,
        result_path=result_path,
        figure_path=figure_path,
        filename=filename,
        verbose=verbose,
    )


def get_completeness_score_network(
    model: nn.Module,
    probes: Dict[str, Dict[str, Dict[str, LogisticRegression]]],
    observations: Observation,
    layer_idx: int,
    concepts: List[str],
    concept_score_method: Literal["soft", "binary", "binary_threshold"],
    epochs: int = 10,
    ignore_layers: List[str] = [],
    verbose: bool = False,
    result_path: str = os.path.join("assets", "results"),
    filename: str = "completeness_score_network.json",
):
    models = {"latest": model}

    observation_zipped, _ = zip_observation_data(observations)
    activations, input, output = compute_activations_from_models(
        models, observation_zipped, ignore_layers
    )
    labels = torch.argmax(output["latest"], dim=1).detach().numpy()
    action_space = len(output["latest"][0])

    # TODO: WHY IS THIS NOT THE SAME AS THE OTHER LABELS IDK
    """
    print(labels)
    print("")
    print(_)
    """

    unique_elements, counts = np.unique(labels, return_counts=True)
    for element, count in zip(unique_elements, counts):
        logging.info(f"Element {element} occurs {count} times.")

    concept_probes = {
        key: list(list(value.values())[-1].values())[layer_idx]
        for key, value in probes.items()
    }

    results = {}
    if "random" in concepts:
        random_probe = {"random": concept_probes["random"]}

        random_loss, random_accuracy = compute_accuracy(
            observation_shape=1,
            action_shape=action_space,
            activations=activations,
            labels=labels,
            probes=random_probe,
            concept_score_method=concept_score_method,
            layer_idx=layer_idx,
            verbose=verbose,
        )
        if verbose:
            logging.info(
                f"Random concept - loss: {random_loss} accuracy: {random_accuracy}"
            )
        random_probes = {}
        positive_observation, test_observation = load_and_split_observation(
            "random", 1.0
        )
        negative_observation, _ = load_and_split_observation("negative_random", 1.0)
        obs_len = len(positive_observation)
        num_concepts = len(probes)
        ratio = obs_len // (num_concepts - 1)
        for i in range(num_concepts - 2):
            pos_obs = positive_observation[i * ratio : (i + 1) * ratio]
            neg_obs = negative_observation[i * ratio : (i + 1) * ratio]
            probe, positive_activations, negative_activations = get_probes(
                models, pos_obs, neg_obs, ignore_layers
            )
            random_probes[f"random{i}"] = list(probe["latest"].values())[layer_idx]

        results = {tuple(["random"]): (random_loss, random_accuracy)}

        for i in range(len(random_probes)):
            probes = list(random_probes.values())[:i]
            loss, accuracy = compute_accuracy(
                observation_shape=1,
                action_shape=action_space,
                activations=activations,
                labels=labels,
                probes=random_probe,
                concept_score_method=concept_score_method,
                layer_idx=layer_idx,
                verbose=verbose,
            )
            results[tuple(random_probes.keys())[: i + 1]] = (loss, accuracy)

    if "random" in concepts:
        concepts.remove("random")

    if len(concepts) > 10:
        logging.warning(
            "The number of concepts is greater than 8. This will take a long time to compute, soooo we will not do it:)."
        )
        return

    for comb in get_combinations(concepts):
        if verbose:
            logging.info(f"\n\n===== Computing accuracy for {comb} =====")
        sub_probes = {
            key: value for key, value in concept_probes.items() if key in comb
        }
        avg_sub_loss = 0
        avg_sub_accuracy = 0
        for _ in range(epochs):
            activations, input, output = compute_activations_from_models(
                models, observation_zipped, ignore_layers
            )
            labels = torch.argmax(output["latest"], dim=1).detach().numpy()
            sub_loss, sub_accuracy = compute_accuracy(
                observation_shape=len(comb),
                action_shape=action_space,
                activations=activations,
                labels=labels,
                probes=sub_probes,
                concept_score_method=concept_score_method,
                layer_idx=layer_idx,
                verbose=verbose,
            )
            avg_sub_loss += sub_loss
            avg_sub_accuracy += sub_accuracy
        avg_sub_loss /= epochs
        avg_sub_accuracy /= epochs
        results[tuple(sorted(comb))] = (avg_sub_loss, avg_sub_accuracy)

    path = os.path.join(result_path, filename)
    write_results(results, path)
    shapley_values = calculate_shapley_values(path, concepts)
    write_results(shapley_values, os.path.join(result_path, "shapley_" + filename))

    return None
