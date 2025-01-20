from typing import List

import torch


# ChatJippity code
class MonteCarloAdvantage:
    def __init__(self, n_workers: int, worker_steps: int, gamma: float):
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.gamma = gamma

    def __call__(
        self,
        done: List[List[bool]],
        rewards: List[List[float]],
        values: List[List[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        advantages = []
        for i in range(self.n_workers):
            worker_advantages = []
            running_return = 0.0

            # Iterate backward over the worker's trajectory
            for t in reversed(range(self.worker_steps)):
                if done[i][t]:  # If the episode ends at t, reset the return
                    running_return = 0.0
                running_return = rewards[i][t] + self.gamma * running_return
                advantage = running_return - values[i][t]
                worker_advantages.append(advantage)

            # Since we iterated backwards, reverse the list to maintain the correct order
            advantages.append(worker_advantages[::-1])

        return advantages
