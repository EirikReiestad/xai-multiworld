import logging
import os
import re
from typing import Dict, Tuple

import numpy as np
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from multiworld.base import MultiWorldEnv
from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from multiworld.multigrid.utils.wrappers import MultiGridConceptObsWrapper
from multiworld.utils.wrappers import ObservationCollectorWrapper
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from utils.common.model_artifact import ModelArtifact
from utils.common.observation import (
    Observation,
    load_and_split_observation,
    zip_observation_data,
)
from utils.core.model_downloader import ModelDownloader
from utils.core.model_loader import ModelLoader
from utils.core.plotting import plot_3d
from xailib.common.activations import compute_activations_from_models
from xailib.common.concept_score import binary_concept_scores
from xailib.common.probes import get_probes
from xailib.common.tcav_score import tcav_scores
from xailib.scripts.completeness_score import write_results


def get_models(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    model = create_model(config, artifact, env, eval=True)

    models = ModelLoader.load_models_from_path("artifacts", model.model)
    return models


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
        activation, input, output = compute_activations_from_models(
            models, observation_zipped, config["layer"]["ignore"]
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
    ignore = config["layer"]["ignore"]
    probes = {}
    positive_activations = {}
    negative_activations = {}

    for concept in config["concepts"]:
        positive_observation = positive_observations[concept]
        negative_observation = negative_observations[concept]
        probe, positive_activations, negative_activations = get_probes(
            models, positive_observation, negative_observation, ignore
        )
        probes[concept] = probe
        positive_activations[concept] = positive_activations
        negative_activations[concept] = negative_activations

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

    logging.info(f"Creating environment {environment_type}.")

    if environment_type:
        return GoToGoalEnv(
            width=width,
            height=height,
            agents=agents,
            preprocessing=preprocessing,
            static=True,
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

    eps_start = 0.0 if eval else 0.95
    eps_end = 0.00 if eval else 0.05

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
