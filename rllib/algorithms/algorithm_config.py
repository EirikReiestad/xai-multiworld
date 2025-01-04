from abc import ABC
from typing import Literal, Optional, Callable
import logging
import gymnasium as gym

from rllib.common.callbacks import RenderingCallback, empty_rendering_callback


class AlgorithmConfig(ABC):
    def __init__(self, algorithm: Literal["DQN", "PPO"]):
        self._algorithm = algorithm
        self._environment: Optional[gym.Env] = None

        self._wandb_project = None
        self._wandb_run_name = None
        self._wandb_reinit = None
        self._wandb_tags = None
        self._wandb_dir = None

        self._rendering_callback = empty_rendering_callback

    def environment(self, env: gym.Env):
        self._environment = env
        return self

    def training(self):
        self._training = True
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
        tags: list[str] = [],
        dir: str = ".",
    ):
        self._wandb_project = project
        self._wandb_run_name = run_name
        self._wandb_reinit = reinit
        self._wandb_tags = tags
        self._wandb_dir = dir
        return self

    def build(self) -> "AlgorithmConfig":
        if self._environment is None:
            raise ValueError("Environment not set.")
        return self
