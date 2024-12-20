from abc import ABC, abstractmethod
from itertools import count
from rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib.core.environment.environment import Environment
from utils.base.wandb import WandB
from multigrid.base import AgentID, ObsType
from multigrid.core.action import Action, int_to_action
from typing import Any, SupportsFloat
import numpy as np


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
        self._episodes_done = 0

    def learn(self, steps: float = np.inf):
        for i in count():
            self.collect_rollouts()

            self.log_episode()
            self.commit_log()

            self._episodes_done += 1

            if self._steps_done >= steps:
                break

    def log_episode(self):
        self.add_log("steps_done", self._steps_done)
        self.add_log("episodes_done", self._episodes_done)

    def collect_rollouts(self):
        observations, _ = self._env.reset()
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
                self.add_log("total_rewards", float(rewards[agent_id]), True)

            for agent_id, value in infos.items():
                for key, value in value.items():
                    self.add_log(key + str(agent_id), value)

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
        rgb_array = self._render()
        self.log_frame(rgb_array)

        return observation, rewards, terminations, truncations, infos

    @abstractmethod
    def predict(self, observation: dict[AgentID, ObsType]) -> dict[AgentID, int]:
        raise NotImplementedError
