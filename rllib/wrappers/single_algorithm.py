from rllib.algorithms.algorithm import Algorithm
from itertools import count
import numpy as np
from multigrid.utils.typing import ObsType


class SingleAlgorithm:
    def __init__(self, algorithm: Algorithm):
        self._algorithm = algorithm

    def learn(self, steps: float = np.inf):
        self._algorithm.collect_rollouts = self.collect_rollouts
        self._algorithm.learn(steps)

    def collect_rollouts(self):
        observation, _ = self._env.reset()
        for t in count():
            self._algorithm._steps += 1
            action = self._get_action(observation)
            next_observation, reward, termination, truncation, info = (
                self._algorithm.step(action)
            )
            self._algorithm.train_step(
                observation,
                next_observation,
                action,
                reward,
                termination,
                truncation,
                info,
            )
            observation = next_observation
            if termination or truncation:
                break

    def _get_action(self, observation: ObsType) -> int:
        return self._algorithm._get_action_from_obs(observation)

    def __getattr__(self, name):
        return getattr(self._algorithm, name)
