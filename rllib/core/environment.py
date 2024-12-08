from abc import ABC
from typing import Literal
from gymnasium import spaces
from rllib.utils.spaces import (
    build_observation_space,
    build_action_space,
    ObservationSpace,
    ActionSpace,
)

from multigrid.base import MultiGridEnv


class Environment(ABC):
    def __init__(
        self,
        env: MultiGridEnv,
    ):
        self._env = env

    @property
    def observation_space(self) -> ObservationSpace:
        return self._observation_spaces[0]

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space[0]

    def _build_environment(self):
        self._env.reset()
        self._build_observation_space()
        self._build_action_space()

    def _build_observation_space(self):
        self._observation_spaces = build_observation_space(self._env.observation_space)
        assert all(
            obs == self._observation_spaces[0] for obs in self._observation_spaces
        ), f"Observation spaces must be the same for all agents, but got {self._observation_spaces}"

    def _build_action_space(self):
        self._action_space = build_action_space(self._env.action_space)
        assert all(
            action == self._action_space[0] for action in self._action_space
        ), f"Action spaces must be the same for all agents, but got {self._action_space}"

    def _render(self):
        self._env.render()
