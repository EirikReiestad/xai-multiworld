from abc import ABC
from itertools import count
from rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib.core.environment import Environment
from rllib.core.wandb import WandB
from multigrid.base import AgentID, ObsType
from typing import Any, SupportsFloat
import numpy as np
import logging


class Algorithm(Environment, WandB, ABC):
    def __init__(self, config: AlgorithmConfig):
        self._config = config.build()
        assert self._config._environment is not None, "Environment not set"  # type: ignore
        Environment.__init__(self, self._config._environment)
        WandB.__init__(
            self,
            self._config._wandb_project,
            self._config._wandb_run_name,
            self._config._wandb_reinit,
            self._config._wandb_tags,
            self._config._wandb_dir,
        )
        self._config = config
        self._build_environment()

    def train(self, steps: float = np.inf):
        for i in count():
            observation, rewards, terminations, truncations, infos = self.step()
            if all(terminations.values()) or all(truncations.values()):
                self._env.reset()
            if i >= steps:
                break

    def step(
        self,
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, SupportsFloat],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        random_action = self._env.action_space.sample()
        observation, rewards, terminations, truncations, infos = self._env.step(
            random_action
        )
        self._render()

        return observation, rewards, terminations, truncations, infos
