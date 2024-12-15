from typing import Any, SupportsFloat

import numpy as np
import torch

from multigrid.core.action import Action
from multigrid.utils.typing import AgentID, ObsType
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.replay_memory import ReplayMemory, Transition
from rllib.algorithms.dqn.utils.preprocessing import preprocess_next_observations
from rllib.core.network.multi_input_network import MultiInputNetwork
from rllib.utils.torch.processing import (
    observation_to_torch,
    observation_to_torch_unsqueeze,
    observations_seperate_to_torch,
    observations_to_torch,
)


class DQN(Algorithm):
    _policy_net: MultiInputNetwork
    _target_net: MultiInputNetwork

    def __init__(self, config: DQNConfig):
        super().__init__(config)
        self._config = config
        self._memory = ReplayMemory(config.replay_buffer_size)
        self._policy_net = MultiInputNetwork(self.observation_space, self.action_space)
        self._target_net = MultiInputNetwork(self.observation_space, self.action_space)
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
                    torch_obs = observation_to_torch_unsqueeze(obs)
                    actions[agent_id] = self._policy_net(*torch_obs).argmax().item()
        else:
            for agent_id in observation.keys():
                actions[agent_id] = np.random.randint(self.action_space.discrete)
        return actions

    def _optimize_model(self):
        if len(self._memory) < self._config.batch_size:
            return

        device = next(self._policy_net.parameters()).device
        transitions = self._memory.sample(self._config.batch_size)

        batch = Transition(*zip(*transitions))

        num_agents = len(batch.state[0].values())

        non_final_mask = []
        for observation in batch.next_state:
            agent_obs = observation.values()
            for obs in agent_obs:
                non_final_mask.append(True if obs is not None else False)

        non_final_next_states = observations_seperate_to_torch(
            batch.next_state, skip_none=True
        )

        state_batch = observations_seperate_to_torch(batch.state)
        action_batch = torch.tensor(
            [torch.tensor(a) for reward in batch.action for a in reward.values()]
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            [torch.tensor(r) for reward in batch.reward for r in reward.values()]
        )

        state_action_values = self._policy_net(*state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self._config.batch_size * num_agents).to(device)

        with torch.no_grad():
            output = self._target_net(*non_final_next_states).max(1).values
            next_state_values[non_final_mask] = output

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

    def _hard_update_target(self):
        if self._steps_done % self._config.target_update != 0:
            return
        print("Updating")
        self._target_net.load_state_dict(self._policy_net.state_dict())

    def _soft_update_target(self):
        target_net_state_dict = self._target_net.state_dict()
        policy_net_state_dict = self._policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * 0.005 + target_net_state_dict[key] * (1 - 0.005)
        self._target_net.load_state_dict(target_net_state_dict)
