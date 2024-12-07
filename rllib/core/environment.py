from abc import ABC
from typing import Literal

from multigrid.base import MultiGridEnv


class Environment(ABC):
    def __init__(
        self,
        env: MultiGridEnv,
    ):
        self._env = env

    def _build_environment(self):
        self._env.reset()

    def _render(self):
        self._env.render()
