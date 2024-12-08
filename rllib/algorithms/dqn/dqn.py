from typing import Any, SupportsFloat

import numpy as np
import torch

from multigrid.core.action import Action
from multigrid.utils.typing import AgentID, ObsType
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.qnetwork import QNetwork
from rllib.algorithms.dqn.replay_buffer import ReplayBuffer
from rllib.algorithms.dqn.utils.preprocessing import preprocess_next_observations


class DQN(Algorithm):
    _policy_net: QNetwork
    _target_net: QNetwork

    def __init__(self, config: DQNConfig):
        super().__init__(config)
        self._replay_buffer = ReplayBuffer(config._replay_buffer_size)

    def train_step(
        self,
        observations: dict[AgentID, ObsType],
        next_observations: dict[AgentID, ObsType],
        actions: dict[AgentID, Action],
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
        truncations: dict[AgentID, bool],
        infos: dict[AgentID, dict[str, Any]],
    ):
        next_obs = preprocess_next_observations(observations)
        self._replay_buffer.add(
            observations, actions, rewards, next_observations, terminations
        )
        self._optimize_model()
        self._hard_update_target()

    def predict(self, observation: dict[AgentID, ObsType]) -> dict[AgentID, int]:
        actions = {}
        for agent_id, obs in observation.items():
            with torch.no_grad():
                actions[agent_id] = self._policy_net(obs).argmax().item()
        return actions

    def _optimize_model(self):
        device = next(self._policy_net.parameters()).device
        batch = self._replay_buffer.sample(self._config._batch_size)

    def _hard_update_target(self):
        self._target_net.load_state_dict(self._policy_net.state_dict())
