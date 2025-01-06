"""
source: https://nn.labml.ai/rl/ppo/gae.html
"""

import numpy as np
from numpy.typing import NDArray


class GAE:
    def __init__(self, n_workers: int, worker_steps: int, gamma: float, lambda_: float):
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, done: NDArray, rewards: NDArray, values: NDArray) -> NDArray:
        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0
        last_mask = 1.0 - done[:, -1]
        last_value = values[:, -1] * last_mask

        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - done[:, t]
            delta = rewards[:, t] + self.gamma * last_value - values[:, t]
            last_advantage = delta + self.gamma * self.lambda_ * last_advantage
            advantages[:, t] = last_advantage
            last_value = values[:, t] * mask

        return advantages
