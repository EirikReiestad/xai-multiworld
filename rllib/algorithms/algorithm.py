from abc import ABC, abstractmethod
from itertools import count
from rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib.core.environment.environment import Environment
from rllib.core.wandb.wandb import WandB
from multigrid.base import AgentID, ObsType
from multigrid.core.action import Action, int_to_action
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

        self._steps_done = 0

    def learn(self, steps: float = np.inf):
        for i in count():
            self.collect_rollouts()

            self.add_log("steps_done", self._steps_done)
            self.commit_log()

            if self._steps_done >= steps:
                break

    def collect_rollouts(self) -> dict:
        observations, _ = self._env.reset()
        total_rewards = {agent_id: 0 for agent_id in observations.keys()}

        for t in count():
            self._steps_done += 1
            actions = self.predict(observations)
            next_observations, rewards, terminations, truncations, infos = self.step(
                actions
            )
            self.train_step(
                observations,
                next_observations,
                actions,
                rewards,
                terminations,
                truncations,
                infos,
            )

            for agent_id in observations.keys():
                self.add_log("total_rewards", total_rewards[agent_id], True)
            observations = next_observations
            if all(terminations.values()) or all(truncations.values()):
                break

    @abstractmethod
    def train_step(
        self,
        observations: dict[AgentID, ObsType],
        next_observations: dict[AgentID, ObsType],
        actions: dict[AgentID, int],
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
        truncations: dict[AgentID, bool],
        infos: dict[AgentID, dict[str, Any]],
    ):
        raise NotImplementedError

    def step(
        self, actions: dict[AgentID, int]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, SupportsFloat],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        observation, rewards, terminations, truncations, infos = self._env.step(actions)
        self._render()

        return observation, rewards, terminations, truncations, infos

    @abstractmethod
    def predict(self, observation: dict[AgentID, ObsType]) -> dict[AgentID, int]:
        raise NotImplementedError
