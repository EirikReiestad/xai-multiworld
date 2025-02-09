import logging
import os
from abc import ABC
from typing import Literal, Optional, Tuple

import gymnasium as gym

from rllib.core.network.network import NetworkType
from utils.common.callbacks import RenderingCallback, empty_rendering_callback


class AlgorithmConfig(ABC):
    def __init__(self, algorithm: Literal["DQN", "PPO"]):
        self._algorithm = algorithm
        self._environment: Optional[gym.Env] = None

        self._wandb_project = None
        self._wandb_run_name = None
        self._wandb_reinit = None
        self._wandb_tags = None
        self._wandb_dir = None

        self._wandb_log_interval = 1
        self._rendering_callback = empty_rendering_callback

        self._training = False
        self._model_path = None
        self.network(network_type=NetworkType.FEED_FORWARD)
        self._eval = False

        self.conv_layers: Tuple[int, ...] = tuple(
            (32, 64, 64),
        )
        self.hidden_units: Tuple[int, ...] = tuple(
            (128, 128),
        )

        self._lr_scheduler = None

    def network(
        self,
        network_type: NetworkType | None = None,
        conv_layers: Tuple[int, ...] | None = None,
        hidden_units: Tuple[int, ...] | None = None,
    ):
        if network_type is not None:
            self._network_type = network_type
        if conv_layers is not None:
            self.conv_layers = conv_layers
        if hidden_units is not None:
            self.hidden_units = hidden_units
        return self

    def environment(self, env: gym.Env):
        self._environment = env
        return self

    def model(self, model: str | None = None):
        if model is not None:
            self._model_path = os.path.join("artifacts", model)
        return self

    def training(
        self,
        eval: bool = False,
        lr_scheduler: Literal["cyclic", "step"] | None = None,
        base_lr: float = 1e-5,
        max_lr: float = 1e-2,
        step_size: int = 1,
        gamma: float = 0.99,
    ):
        self._training = True
        self._eval = eval
        self._lr_scheduler = lr_scheduler
        self._base_lr = base_lr
        self._max_lr = max_lr
        self._scheduler_step_size = step_size
        self._scheduler_gamma = gamma
        return self

    def debugging(
        self,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        | None = None,
    ):
        self._debugging = log_level
        logging.basicConfig(level=log_level)
        return self

    def rendering(
        self,
        rendering: bool = True,
        callback: RenderingCallback = empty_rendering_callback,
    ):
        self._rendering = rendering
        self._rendering_callback = callback
        return self

    def wandb(
        self,
        project: str,
        run_name: str | None = None,
        reinit: bool = True,
        log_interval: int = 1,
        tags: list[str] = [],
        dir: str = ".",
    ):
        self._wandb_project = project
        self._wandb_run_name = run_name
        self._wandb_reinit = reinit
        self._wandb_tags = tags
        self._wandb_dir = dir
        self._wandb_log_interval = log_interval
        return self

    def build(self) -> "AlgorithmConfig":
        if self._environment is None:
            raise ValueError("Environment not set.")
        return self
