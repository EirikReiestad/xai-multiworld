from rllib.algorithms.algorithm import Algorithm
from itertools import count
import numpy as np


class SingleAlgorithm:
    def __init__(self, algorithm: Algorithm):
        self._algorithm = algorithm

    def learn(self, steps: float = np.inf):
        for i in count():
            self._env.reset()
            self.collect_rollouts()
            if self._steps >= steps:
                break

    def collect_rollouts(self):
        for t in count():
            self._algorithm._steps += 1
            observation, reward, termination, truncation, info = self.step()
            if termination or truncation:
                break

    def __getattr__(self, name):
        return getattr(self._algorithm, name)
