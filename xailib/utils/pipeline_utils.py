import logging
import os
import re
from typing import Dict

from torch import Value

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
from utils.core.model_downloader import ModelDownloader


def download_models(config: Dict, force: bool = False):
    models = [
        f"model_{i}:latest"
        for i in range(
            config["wandb"]["models"]["low"],
            config["wandb"]["models"]["high"],
            config["wandb"]["models"]["step"],
        )
    ]
    model_folder = os.path.join(config["path"]["artifacts"])

    if force is False:
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


def create_environment(config: Dict, artifact: ModelArtifact):
    height = artifact.metadata.get("height") or 10
    width = artifact.metadata.get("width") or 10
    agents = artifact.metadata.get("agents") or 1
    environment_type = artifact.metadata.get("environment_type") or "go-to-goal"
    preprocessing = artifact.metadata.get("preprocessing") or "none"

    preprocessing = PreprocessingEnum(preprocessing)

    logging.info(f"Creating environment {environment_type}.")

    if environment_type:
        return GoToGoalEnv(
            width=width, height=height, agents=agents, preprocessing=preprocessing
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
