"""
source: https://nn.labml.ai/rl/ppo/gae.html
"""

import torch
from typing import List


class GAE:
    def __init__(self, n_workers: int, worker_steps: int, gamma: float, lambda_: float):
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(
        self,
        done: List[List[bool]],
        rewards: List[List[float]],
        values: List[List[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        advantages = []
        last_advantage: List[torch.Tensor] = [
            torch.tensor(0.0) for _ in range(self.n_workers)
        ]
        mask = [1.0 - d[-1] for d in done]
        last_value = [v[-1] * mask[i] for i, v in enumerate(values)]

        for t in reversed(range(self.worker_steps)):
            mask = [1.0 - d[t] for d in done]
            delta = [
                rewards[i][t] + self.gamma * last_value[i] - values[i][t]
                for i in range(self.n_workers)
            ]
            for i in range(self.n_workers):
                last_advantage[i] = (
                    delta[i] + self.gamma * self.lambda_ * last_advantage[i]
                )
            advantages.append(last_advantage.copy())
            last_value = [values[i][t].clone() * mask[i] for i in range(self.n_workers)]
        advantages_transposed = list(zip(*advantages))
        return advantages_transposed
