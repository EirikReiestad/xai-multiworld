import logging
import os
import re
from typing import Dict, Literal

import torch.nn as nn

from multiworld.base import MultiWorldEnv
from multiworld.multigrid.utils.wrappers import MultiGridConceptObsWrapper
from multiworld.utils.wrappers import ObservationCollectorWrapper
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from utils.common.model_artifact import ModelArtifact
from utils.core.model_downloader import ModelDownloader
from utils.core.model_loader import ModelLoader


def get_models(
    artifact: ModelArtifact,
    model_type: Literal["dqn"],
    env: MultiWorldEnv,
    eval: bool,
    artifact_path: str = "artifacts",
):
    model = create_model(artifact, model_type, artifact_path, env, eval)

    models = ModelLoader.load_models_from_path("artifacts", model.model)
    return models


def get_newest_model(path: str):
    model_dirs = os.listdir(path)
    sorted_model_dirs = sorted(
        model_dirs, key=lambda x: int(re.search(r"\d+", x).group())
    )
    latest_model_dir = sorted_model_dirs[-1]
    return latest_model_dir


def get_latest_model(
    artifact: ModelArtifact,
    model_type: Literal["dqn"],
    artifact_path: str,
    env: MultiWorldEnv,
    eval: bool,
) -> nn.Module:
    model = create_model(artifact, model_type, artifact_path, env, eval)

    model = ModelLoader.load_latest_model_from_path("artifacts", model.model)
    return model


def create_model(
    artifact: ModelArtifact,
    model_type: Literal["dqn"],
    artifact_path: str,
    env: MultiWorldEnv | ObservationCollectorWrapper | MultiGridConceptObsWrapper,
    eval: bool = False,
) -> Algorithm:
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
            .model(get_newest_model(artifact_path))
            .debugging(log_level="INFO")
            .environment(env=env)
        )

        dqn = DQN(dqn_config)
        return dqn

    raise ValueError(f"Sorry but model type of {model_type} is not supported.")


def download_models(
    low: int,
    high: int,
    step: int,
    model_name: str,
    wandb_project_folder: str,
    artifact_path: str = os.path.join("artifacts"),
    force_update: bool = False,
):
    models = [f"{model_name}_{i}:latest" for i in range(low, high, step)]

    if force_update is False:
        try:
            artifacts = [
                model_name.split(":")[0] for model_name in os.listdir(artifact_path)
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
        project_folder=wandb_project_folder,
        models=models,
        model_folder=artifact_path,
    )
    model_downloader.download()
