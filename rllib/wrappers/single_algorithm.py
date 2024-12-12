from itertools import count
from typing import Any, SupportsFloat

import numpy as np
import torch

from multigrid.utils.typing import AgentID, ObsType
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.dqn.replay_memory import Transition


class SingleAlgorithm:
    def __init__(self, algorithm: Algorithm):
        self._algorithm = algorithm

    def learn(self, steps: float = np.inf):
        self._algorithm.collect_rollouts = self.collect_rollouts
        self._algorithm.step = lambda action: self.step(action)
        self._algorithm._optimize_model = self._optimize_model
        self._algorithm.learn(steps)

    def collect_rollouts(self):
        observation, _ = self._env.reset()
        observations: dict[AgentID, ObsType] = {"0": observation}
        total_reward = 0
        for t in count():
            self._algorithm._steps_done += 1
            action = self._algorithm.predict(observations)
            next_observation, reward, termination, truncation, info = (
                self._algorithm.step(action)
            )
            total_reward += reward
            actions = {"0": action}
            self._algorithm.train_step(
                observations,
                next_observation,
                actions,
                reward,
                termination,
                truncation,
                info,
            )
            observation = next_observation
            if termination or truncation:
                break

        self.log({"total_reward": total_reward})

    def step(
        self, action: dict[AgentID, int]
    ) -> tuple[
        ObsType,
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        action = next(iter(action.values()))
        observation, reward, termination, truncation, info = self._env.step(action)
        self._render()

        return observation, reward, termination, truncation, info

    def _optimize_model(self):
        if len(self._memory) < self._config.batch_size:
            return

        device = next(self._policy_net.parameters()).device
        batch = self._memory.sample(self._config.batch_size)

        batch = Transition(*zip(*batch))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.stack(
            [torch.from_numpy(s) for s in batch.next_state if s is not None]
        )

        state_batch = [torch.tensor(state["0"]) for state in batch.state]
        action_batch = torch.tensor(
            [torch.tensor(action["0"]) for action in batch.action]
        )
        reward_batch = torch.tensor(
            [torch.tensor(reward["0"]) for reward in batch.reward]
        )

        state_action_values = self._policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self._config.batch_size).to(device)

        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self._target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = (
            next_state_values * self._config.gamma + reward_batch
        )
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        ).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def __getattr__(self, name):
        return getattr(self._algorithm, name)
