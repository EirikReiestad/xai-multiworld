import logging
import os
from typing import Dict, List, Literal

from sklearn.linear_model import LinearRegression

from multiworld.base import MultiWorldEnv
from rllib.algorithms.algorithm import Algorithm
from utils.common.collections import get_combinations
from utils.common.model import create_model
from utils.common.model_artifact import ModelArtifact
from utils.common.observation import Observation, zip_observation_data
from utils.common.write import write_results
from xailib.common.activations import compute_activations_from_models
from xailib.utils.metrics import calculate_shapley_values, compute_accuracy


def get_completeness_score(
    probes: Dict[str, Dict[str, Dict[str, LinearRegression]]],
    concepts: List[str],
    artifact: ModelArtifact,
    environment: MultiWorldEnv,
    observations: Observation,
    layer_idx: int,
    ignore_layers: List = [],
    model_type: Literal["dqn"] = "dqn",
    method: Literal["decisiontree", "network"] = "network",
    artifact_path: str = "artifacts",
    eval: bool = True,
    verbose: bool = False,
):
    model = create_model(artifact, model_type, artifact_path, environment, eval)
    if method == "network":
        get_completeness_score_network(
            model,
            probes,
            observations,
            layer_idx,
            concepts,
            ignore_layers=ignore_layers,
            verbose=verbose,
        )
    elif method == "decisiontree":
        get_completeness_score_decision_tree(
            model, probes, artifacts, environment, observations, verbose
        )


def get_completeness_score_decision_tree(
    model: Algorithm,
    probes: Dict[str, Dict[str, Dict[str, LinearRegression]]],
    artifacts: ModelArtifact,
    environment: MultiWorldEnv,
    observations: Observation,
    verbose: bool = False,
):
    layer_idx = config["analyze"]["layer_idx"]
    ignore = config["analyze"]["ignore_layers"]
    concepts = config["concepts"].copy()

    model = create_model(config, artifacts, environment, eval=True)
    action_space = int(model.action_space.n)
    models = {"latest": model.model}

    observation_zipped = zip_observation_data(observations)
    activations, input, output = compute_activations_from_models(
        models, observation_zipped, ignore
    )
    probes = {
        key: list(list(value.values())[-1].values())[layer_idx]
        for key, value in probes.items()
    }

    compute_accuracy_decision_tree(
        len(concepts),
        action_space,
        activations,
        observations,
        probes,
        layer_idx,
        concepts,
        verbose=verbose,
    )


def get_completeness_score_network(
    model: Algorithm,
    probes: Dict[str, Dict[str, Dict[str, LinearRegression]]],
    observations: Observation,
    layer_idx: int,
    concepts: List[str],
    ignore_layers: List[str] = [],
    verbose: bool = False,
):
    action_space = int(model.action_space.n)
    models = {"latest": model.model}

    observation_zipped = zip_observation_data(observations)
    activations, input, output = compute_activations_from_models(
        models, observation_zipped, ignore_layers
    )

    probes = {
        key: list(list(value.values())[-1].values())[layer_idx]
        for key, value in probes.items()
    }

    random_probe = {"random": probes["random"]}

    random_loss, random_accuracy = compute_accuracy(
        1,
        action_space,
        activations,
        observations,
        random_probe,
        layer_idx,
        verbose=verbose,
    )
    if verbose:
        logging.info(
            f"Random concept - loss: {random_loss} accuracy: {random_accuracy}"
        )

    """
    if "random" in concepts:
        concepts.remove("random")
    """

    results = {tuple(["random"]): (random_loss, random_accuracy)}
    for comb in get_combinations(concepts):
        if verbose:
            logging.info(f"\n\n===== Computing accuracy for {comb} =====")
        sub_probes = {key: value for key, value in probes.items() if key in comb}
        sub_loss, sub_accuracy = compute_accuracy(
            len(comb),
            action_space,
            activations,
            observations,
            sub_probes,
            layer_idx,
            verbose=verbose,
        )
        results[tuple(sorted(comb))] = (sub_loss, sub_accuracy)

    path = os.path.join(
        "assets", "results", f"concept_combination_accuracies_{layer_idx}.json"
    )

    write_results(results, path)
    calculate_shapley_values(path, concepts)
