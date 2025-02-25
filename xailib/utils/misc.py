import ast
import json
import logging
import os
import re
from typing import Dict, Literal, Tuple

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
from xailib.common.activations import compute_activations_from_models
from xailib.common.probes import get_probes


def read_results(path: str) -> Dict:
    with open(path, "r") as f:
        results = json.load(f)
        results = {ast.literal_eval(key): value for key, value in results.items()}
    return results


def write_results(results: Dict, path: str):
    with open(path, "w") as f:
        results = {str(key): value for key, value in results.items()}
        json.dump(results, f)


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
