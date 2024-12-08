from typing import Any, SupportsFloat

import numpy as np
import torch

from multigrid.core.action import Action
from multigrid.utils.typing import AgentID, ObsType
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.qnetwork import QNetwork
from rllib.algorithms.dqn.replay_memory import ReplayMemory, Transition
from rllib.algorithms.dqn.utils.preprocessing import preprocess_next_observations


class DQN(Algorithm):
    _policy_net: QNetwork
    _target_net: QNetwork

    def __init__(self, config: DQNConfig):
        super().__init__(config)
        self._config = config
        self._memory = ReplayMemory(config.replay_buffer_size)
        self._policy_net = QNetwork(self.observation_space, self.action_space)
        self._target_net = QNetwork(self.observation_space, self.action_space)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._optimizer = torch.optim.AdamW(
            self._policy_net.parameters(), lr=config.learning_rate, amsgrad=True
        )

    def train_step(
        self,
        observations: dict[AgentID, ObsType],
        next_observations: dict[AgentID, ObsType],
        actions: dict[AgentID, int],
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
        truncations: dict[AgentID, bool],
        infos: dict[AgentID, dict[str, Any]],
    ):
        next_obs = preprocess_next_observations(
            next_observations, terminations, truncations
        )
        self._memory.add(observations, actions, rewards, next_obs)
        self._optimize_model()
        self._soft_update_target()
        self._hard_update_target()

    def predict(self, observation: dict[AgentID, ObsType]) -> dict[AgentID, int]:
        sample = np.random.rand()
        eps_threshold = self._config.eps_end + (
            self._config.eps_start - self._config.eps_end
        ) * np.exp(-1.0 * self._steps_done / self._config.eps_decay)
        actions = {}
        if sample > eps_threshold:
            for agent_id, obs in observation.items():
                with torch.no_grad():
                    torch_obs = torch.from_numpy(obs).unsqueeze(dim=0)
                    actions[agent_id] = self._policy_net(torch_obs).argmax().item()
        else:
            for agent_id in observation.keys():
                actions[agent_id] = np.random.randint(self.action_space.discrete)
        return actions

    def _optimize_model(self):
        if len(self._memory) < self._config.batch_size:
            return

        device = next(self._policy_net.parameters()).device
        batch = self._memory.sample(self._config.batch_size)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [torch.from_numpy(s) for s in next_states if s is not None]
        )

        states = torch.stack([torch.from_numpy(s) for s in states])
        state_action_values = self._policy_net(states).gather(1, actions)
        next_state_values = torch.zeros(self._config.batch_size).to(device)

        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self._target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = next_state_values * self._config.gamma + rewards
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        ).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _hard_update_target(self):
        if self._steps_done % self._config.target_update != 0:
            return
        self._target_net.load_state_dict(self._policy_net.state_dict())

    def _soft_update_target(self):
        target_net_state_dict = self._target_net.state_dict()
        policy_net_state_dict = self._policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * 0.005 + target_net_state_dict[key] * (1 - 0.005)
        self._target_net.load_state_dict(target_net_state_dict)
