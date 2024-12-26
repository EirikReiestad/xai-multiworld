from abc import ABC

from multigrid.wrappers import ConceptObsWrapper
from rllib.algorithms.algorithm import Algorithm, AlgorithmConfig


class ConceptBottleneck(ABC):
    def __init__(self, algorithm: Algorithm):
        super().__init__()

        if type(self._algorithm._env) is not ConceptObsWrapper:
            raise ValueError("Environment must be a ConceptObsWrapper")

        self._algorithm = algorithm

    def __getattr__(self, name):
        return getattr(self._algorithm, name)
