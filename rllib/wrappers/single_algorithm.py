from rllib.algorithms.algorithm import Algorithm
from itertools import count
import numpy as np


class SingleAlgorithm:
    def __init__(self, algorithm: Algorithm):
        self._algorithm = algorithm

    def train(self, steps: float = np.inf):
        for i in count():
            observation, reward, termination, truncation, info = self.step()
            if termination or truncation:
                self._env.reset()
            if i >= steps:
                break

    def __getattr__(self, name):
        return getattr(self._algorithm, name)
