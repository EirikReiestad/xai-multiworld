import ast
import json
import logging
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from torch.utils.data import TensorDataset

from multiworld.base import MultiWorldEnv
from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from multiworld.multigrid.utils.wrappers import MultiGridConceptObsWrapper
from multiworld.utils.wrappers import ObservationCollectorWrapper
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from utils.common.collections import get_combinations
from utils.common.model_artifact import ModelArtifact
from utils.common.numpy_collections import convert_numpy_to_float
from utils.common.observation import (
    Observation,
    load_and_split_observation,
    zip_observation_data,
)
from utils.core.model_downloader import ModelDownloader
from utils.core.model_loader import ModelLoader
from utils.core.plotting import plot_3d
from xailib.common.activations import compute_activations_from_models
from xailib.common.concept_score import (
    binary_concept_scores,
    individual_binary_concept_score,
    individual_soft_concept_score,
)
from xailib.common.probes import get_probes
from xailib.common.tcav_score import tcav_scores
from xailib.common.train_model import train_model
from xailib.core.network.feed_forward import FeedForwardNetwork


def calculate_probe_robustness(config: Dict, model: nn.Module):
    layer_idx = config["analyze"]["layer_idx"]
    concepts = config["concepts"]
    splits = config["analyze"]["splits"]
    epochs = config["analyze"]["robustness_epochs"]

    concept_similarities = defaultdict(lambda: defaultdict(float))
    for _ in range(epochs):
        concept_probe_robustness = probe_robustness(concepts, layer_idx, model, splits)
        for concept in concepts:
            for key, value in concept_probe_robustness[concept].items():
                concept_similarities[concept][key] += value

    for concept in concepts:
        for key in concept_similarities[concept].keys():
            concept_similarities[concept][key] /= epochs

    path = os.path.join(config["path"]["results"], "probe_robustness.json")
    concept_similarities = convert_numpy_to_float(concept_similarities)
    write_results(concept_similarities, path)
    log_similarity(concept_similarities, concepts)

    return concept_similarities


def probe_robustness(concepts, layer_idx, model, splits) -> Dict:
    concept_similarities = {}
    for concept in concepts:
        similarities = get_probe_similarity(concept, layer_idx, model, splits)
        concept_similarities[concept] = similarities

    return concept_similarities


def log_similarity(similarities: Dict, concepts: List[str]):
    y_axis_values = sorted(next(iter(similarities.values())).keys(), reverse=True)

    table_data = []
    for y in y_axis_values:
        row = [y] + [similarities[key][y][0][0] for key in similarities.keys()]
        table_data.append(row)

    headers = ["dataset size"] + concepts
    logging.info("\n" + tabulate(table_data, headers=headers))


def get_probe_similarity(
    concept: str, layer_idx: int, model: nn.Module, splits: List[float]
) -> Dict[float, float]:
    base = get_probe(concept, layer_idx, model, 1.0)
    if base is None:
        return {}

    probe_splits = {}
    for split in splits:
        probe = get_probe(concept, layer_idx, model, split)
        if probe is None:
            continue
        probe_splits[split] = probe

    similarities = {
        key: cosine_similarity(base.coef_, probe.coef_)
        for key, probe in probe_splits.items()
    }
    return similarities


def get_probe(concept: str, layer_idx: int, model: nn.Module, split: float = 0.8):
    ignore = []

    model = ModelLoader.load_latest_model_from_path("artifacts", model)
    models = {"latest": model}
    positive_observation, test_observation = load_and_split_observation(concept, split)
    negative_observation, _ = load_and_split_observation("negative_" + concept, split)

    if len(positive_observation) == 0:
        logging.warning(f"Positive observation for {concept} is empty.")
        return None

    probes, positive_activations, negative_activations = get_probes(
        models, positive_observation, negative_observation, ignore
    )

    probe = list(probes["latest"].values())[layer_idx]

    return probe


def calculate_statistics(
    config: Dict, activations: Dict[str, Dict[str, Dict[str, np.ndarray]]]
):
    concepts = config["concepts"]
    stats = {}
    for concept in concepts:
        latest_activations = {"latest": list(activations[concept].values())[-1]}
        stat = calculate_statistic(config, latest_activations)
        stats[concept] = stat

    path = os.path.join(config["path"]["results"], "probe_statistics.json")
    stats = convert_numpy_to_float(stats)
    write_results(stats, path)
    log_stats(stats)


def calculate_statistic(
    config: Dict,
    activations: Dict[str, Dict[str, np.ndarray]],
):
    layer_idx = config["analyze"]["layer_idx"]
    layer_activations = list(activations["latest"].values())[layer_idx]["output"]
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

    pairwise_distances = pdist(points, "euclidean")
    mean_distance = np.mean(pairwise_distances)
    density = 1 / mean_distance if mean_distance != 0 else float("inf")

    stats = {
        # "mean": float(mean),
        # "variance": float(variance),
        # "std_dev": float(std_dev),
        # "median": float(median),
        # "range": float(range_val),
        # "iq_range": float(iq_range),
        "density": density,
    }
    return stats


def log_stats(stats: Dict[str, Dict[str, float]]):
    metrics = list(next(iter(stats.values())).keys())
    concepts = list(stats.keys())

    table_data = []
    for metric in metrics:
        row = [metric] + [stats[concept][metric] for concept in concepts]
        table_data.append(row)

    table = tabulate(
        table_data,
        headers=["Metric"] + concepts,
    )
    logging.info(f"\n{table}")


def get_completeness_score(
    config: Dict,
    probes: Dict[str, Dict[str, Dict[str, LinearRegression]]],
    artifacts: ModelArtifact,
    environment: MultiWorldEnv,
    observations: Observation,
    verbose: bool = False,
):
    layer_idx = config["analyze"]["layer_idx"]
    ignore = config["analyze"]["ignore_layers"]

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

    concepts = config["concepts"].copy()

    if "random" in concepts:
        concepts.remove("random")

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


def read_results(path: str) -> Dict:
    with open(path, "r") as f:
        results = json.load(f)
        results = {ast.literal_eval(key): value for key, value in results.items()}
    return results


def write_results(results: Dict, path: str):
    with open(path, "w") as f:
        results = {str(key): value for key, value in results.items()}
        json.dump(results, f)


def calculate_shapley_values(path: str, concepts: List[str]):
    results = read_results(path)

    table = sorted(
        [(comb, loss, accuracy) for comb, (loss, accuracy) in results.items()],
        key=lambda x: x[2],
        reverse=True,
    )
    logging.info("\n" + tabulate(table, headers=["Combination", "Loss", "Accuracy"]))

    shapley_values = defaultdict(float)
    N = len(concepts)
    for concept in concepts:
        other_concepts = concepts.copy()
        other_concepts.remove(concept)
        for comb in get_combinations(other_concepts):
            comb_u_concept = tuple(sorted(comb + [concept]))
            comb = tuple(comb)
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
    return shapley_values


def compute_accuracy(
    observation_shape: int,
    action_shape: int,
    activations: Dict[str, Dict],
    observations: Observation,
    probes: Dict[str, LogisticRegression],
    layer_idx: int,
    verbose: bool = False,
):
    hidden_units = 500

    epochs = 3
    batch_size = 64
    learning_rate = 0.001
    val_split = 0.2
    test_split = 0.1

    concept_scores = np.array(
        get_concept_score(
            activations, probes, layer_idx, concept_score_method="binary"
        ),
        dtype=np.float32,
    )
    labels = np.array(observations[..., Observation.LABEL], dtype=np.float32)

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


def get_models(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    model = create_model(config, artifact, env, eval=True)

    models = ModelLoader.load_models_from_path("artifacts", model.model)
    return models


def get_latest_model(
    config: Dict, env: MultiWorldEnv, artifact: ModelArtifact
) -> nn.Module:
    model = create_model(config, artifact, env, eval=True)

    model = ModelLoader.load_latest_model_from_path("artifacts", model.model)
    return model


def get_tcav_scores(
    config: Dict,
    test_activations: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    test_output: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    probes: Dict[str, Dict[str, Dict[str, LinearRegression]]],
) -> Dict[str, Dict[str, float]]:
    concept_tcav_scores = {}

    for concept in config["concepts"]:
        scores = tcav_scores(
            test_activations[concept], test_output[concept], probes[concept]
        )
        concept_tcav_scores[concept] = scores

        plot_3d(
            scores,
            label=concept,
            filename="tcav_" + concept,
            min=0,
            max=1,
            show=False,
        )
    write_results(
        concept_tcav_scores,
        os.path.join(config["path"]["results"], "tcav_scores.json"),
    )
    return concept_tcav_scores


def get_concept_scores(
    config: Dict,
    test_activations: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    probes: Dict[str, Dict[str, Dict[str, LinearRegression]]],
) -> Dict[str, Dict[str, float]]:
    concept_scores = {}

    for concept in config["concepts"]:
        concept_score = binary_concept_scores(
            test_activations[concept], probes[concept]
        )
        concept_scores[concept] = concept_score

        plot_3d(
            concept_score,
            label=concept,
            filename=f"concept_score_{concept}",
            min=0,
            max=1,
            show=False,
        )
    write_results(
        concept_scores, os.path.join(config["path"]["results"], "concept_scores.json")
    )
    return concept_scores


def get_observations(
    config: Dict,
) -> Tuple[
    Dict[str, Observation],
    Dict[str, Observation],
    Dict[str, Observation],
    Dict[str, Observation],
]:
    concepts = config["concepts"]

    positive_observations = {}
    negative_observations = {}
    test_positive_observations = {}
    test_negative_observations = {}

    for concept in concepts:
        positive_observation, test_observation = load_and_split_observation(
            concept, 0.8
        )
        negative_observation, _ = load_and_split_observation("negative_" + concept, 0.8)

        positive_observations[concept] = positive_observation
        negative_observations[concept] = negative_observation
        test_positive_observations[concept] = test_observation
        test_negative_observations[concept] = negative_observation

    return (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    )


def get_activations(
    config: Dict, models: Dict[str, nn.Module], observations: Observation
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
]:
    ignore = config["analyze"]["ignore_layers"]
    observation_zipped = zip_observation_data(observations)
    activations, input, output = compute_activations_from_models(
        models, observation_zipped, ignore
    )
    return activations, input, output


def get_concept_activations(
    config: Dict, observation: Dict[str, Observation], models: Dict[str, nn.Module]
) -> Tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
]:
    activations = {}
    inputs = {}
    outputs = {}

    for concept in config["concepts"]:
        observation_zipped = zip_observation_data(observation[concept])
        ignore = config["analyze"]["ignore_layers"]
        activation, input, output = compute_activations_from_models(
            models, observation_zipped, ignore
        )
        activations[concept] = activation
        inputs[concept] = input
        outputs[concept] = output
    return activations, inputs, outputs


def get_probes_and_activations(
    config: Dict,
    models: Dict[str, nn.Module],
    positive_observations: Dict[str, Observation],
    negative_observations: Dict[str, Observation],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, LinearRegression]]],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    ignore = config["analyze"]["ignore_layers"]
    probes = {}
    positive_activations = {}
    negative_activations = {}

    for concept in config["concepts"]:
        positive_observation = positive_observations[concept]
        negative_observation = negative_observations[concept]
        probe, positive_activation, negative_activation = get_probes(
            models, positive_observation, negative_observation, ignore
        )
        probes[concept] = probe
        positive_activations[concept] = positive_activation
        negative_activations[concept] = negative_activation

    return probes, positive_activations, negative_activations


def download_models(config: Dict):
    models = [
        f"model_{i}:latest"
        for i in range(
            config["wandb"]["models"]["low"],
            config["wandb"]["models"]["high"],
            config["wandb"]["models"]["step"],
        )
    ]
    model_folder = os.path.join(config["path"]["artifacts"])

    if config["force_update"] is False:
        try:
            artifacts = [
                model_name.split(":")[0] for model_name in os.listdir(model_folder)
            ]
            model_names = [model_name.split(":")[0] for model_name in models]

            if sorted(artifacts) == sorted(model_names):
                logging.info(
                    "Artifacts already exists, so we do not need to download them:)"
                )
                return
        except FileNotFoundError:
            pass

    model_downloader = ModelDownloader(
        project_folder=config["wandb"]["project_folder"],
        models=models,
        model_folder=model_folder,
    )
    model_downloader.download()


def create_environment(artifact: ModelArtifact):
    height = artifact.metadata.get("height") or 10
    width = artifact.metadata.get("width") or 10
    agents = artifact.metadata.get("agents") or 1
    environment_type = artifact.metadata.get("environment_type") or "go-to-goal"
    preprocessing = artifact.metadata.get("preprocessing") or "none"

    preprocessing = PreprocessingEnum(preprocessing)

    if environment_type:
        return GoToGoalEnv(
            width=width,
            height=height,
            agents=agents,
            preprocessing=preprocessing,
            static=True,
            render_mode="rgb_array",
        )
    else:
        raise ValueError(
            f"Sorry but environment type of {environment_type} is not supported."
        )


def create_model(
    config: Dict,
    artifact: ModelArtifact,
    env: MultiWorldEnv | ObservationCollectorWrapper | MultiGridConceptObsWrapper,
    eval: bool = False,
) -> Algorithm:
    model_type = config["model"]["type"]

    conv_layers = artifact.metadata.get("conv_layers")
    hidden_units = artifact.metadata.get("hidden_units")
    network_type = artifact.metadata.get("network_type") or "feed_forward"
    network_type = NetworkType(network_type.lower())

    eps_start = 0.0 if eval else 1
    eps_end = 0.00 if eval else 1

    if model_type == "dqn":
        dqn_config = (
            DQNConfig(
                eps_start=eps_start,
                eps_end=eps_end,
            )
            .network(
                network_type=network_type,
                conv_layers=conv_layers,
                hidden_units=hidden_units,
            )
            .model(get_newest_model(config["path"]["artifacts"]))
            .debugging(log_level="INFO")
            .environment(env=env)
        )

        dqn = DQN(dqn_config)
        return dqn

    raise ValueError(f"Sorry but model type of {model_type} is not supported.")


def get_newest_model(path: str):
    model_dirs = os.listdir(path)
    sorted_model_dirs = sorted(
        model_dirs, key=lambda x: int(re.search(r"\d+", x).group())
    )
    latest_model_dir = sorted_model_dirs[-1]
    return latest_model_dir
