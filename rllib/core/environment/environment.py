from abc import ABC
import numpy as np
from typing import Optional
from gymnasium import spaces
from rllib.utils.spaces import (
    build_observation_space,
    build_action_space,
    ObservationSpace,
    ActionSpace,
)
from rllib.utils.validation import all_same_values_in_dict

from multiworld.multigrid.base import MultiGridEnv
from multiworld.swarm.base import SwarmEnv


class Environment(ABC):
    def __init__(
        self,
        env: MultiGridEnv | SwarmEnv,
    ):
        self._env = env

    @property
    def observation_space(self) -> ObservationSpace:
        return next(iter(self._observation_spaces.values()))

    @property
    def action_space(self) -> ActionSpace:
        return next(iter(self._action_space.values()))

    def _build_environment(self):
        self._env.reset()
        self._build_observation_spaces()
        self._build_action_space()

    def _build_observation_spaces(self):
        """
        The observation space from the environment should give a dict with a key (index to the agent) and a value (the observation space for the agent)
        """
        assert isinstance(
            self._env.observation_space, spaces.Dict
        ), f"Observation space must be a Dict, but got {self._env.observation_space}"
        self._observation_spaces = {
            agent_index: build_observation_space(observation_space)
            for agent_index, observation_space in self._env.observation_space.items()
        }
        assert all_same_values_in_dict(
            self._observation_spaces
        ), f"Observation spaces must be the same for all agents, but got {self._observation_spaces}"

    def _build_action_space(self):
        """
        The action space from the environment should give a dict with a key (index to the agent) and a value (the action space for the agent)
        """
        assert isinstance(
            self._env.action_space, spaces.Dict
        ), f"Action space must be a Dict, but got {self._env.action_space}"
        self._action_space = {
            agent_index: build_action_space(action_space)
            for agent_index, action_space in self._env.action_space.items()
        }
        assert all_same_values_in_dict(
            self._action_space
        ), f"Action spaces must be the same for all agents, but got {self._action_space}"

    def _render(self) -> Optional[np.ndarray]:
        return self._env.render()
