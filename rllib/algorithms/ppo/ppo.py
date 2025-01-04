from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.ppo.ppo_config import PPOConfig
from rllib.utils.dqn.preprocessing import preprocess_next_observations
from multigrid.utils.typing import AgentID, ObsType
from typing import SupportsFloat, Mapping, Any, List, Dict, Tuple
from rllib.core.network.actor_critic_multi_input_network import (
    ActorCriticMultiInputNetwork,
)
from multigrid.core.action import Action
import torch
import numpy as np
from numpy.typing import NDArray
from rllib.utils.torch.processing import (
    observation_to_torch_unsqueeze,
    observations_seperate_to_torch,
)
import torch.nn as nn
from rllib.core.memory.trajectory_buffer import TrajectoryBuffer, Trajectory
from rllib.utils.ppo.calculations import compute_advantages, compute_log_probs, ppo_loss
from rllib.utils.dqn.misc import get_non_final_mask

"""
PPO Paper: https://arxiv.org/abs/1707.06347
"""


class PPO(Algorithm):
    _policy_net: ActorCriticMultiInputNetwork

    def __init__(self, config: PPOConfig):
        super().__init__(config)
        self._config = config
        self._policy_net = ActorCriticMultiInputNetwork(
            self.observation_space, self.action_space
        )

        self._action_probs = {}
        self._trajectory_buffer = TrajectoryBuffer(self._config.target_update)

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
        self._trajectory_buffer.add_dict(
            keys=observations.keys(),
            state=observations,
            action=actions,
            action_prob=self._action_probs,
            reward=rewards,
        )
        self._optimize_model()

    def log_episode(self):
        super().log_episode()
        self.log_model(self._policy_net, f"model_{self._episodes_done}")

    def predict(self, observation: dict[AgentID, ObsType]) -> Dict[AgentID, int]:
        actions, action_probs = self._get_actions(observation)
        self._action_probs = action_probs
        return actions

    def load_model(self, model: Mapping[str, Any]):
        self._policy_net.load_state_dict(model)
        self._policy_net.eval()

    @property
    def model(self) -> nn.Module:
        return self._policy_net

    def _get_actions(
        self, observations: Dict[AgentID, ObsType]
    ) -> Tuple[Dict[AgentID, int], Dict[AgentID, NDArray]]:
        actions = {}
        action_probs = {}
        for agent_id, obs in observations.items():
            actions[agent_id], action_probs[agent_id] = self._get_action(obs)
        return actions, action_probs

    def _get_action(self, observation: ObsType) -> Tuple[Action, NDArray]:
        with torch.no_grad():
            torch_obs = observation_to_torch_unsqueeze(observation)
            action_prob, _ = self._policy_net(*torch_obs)
            action_prob = action_prob.squeeze().detach().numpy()
            action = np.random.choice(range(self.action_space.discrete), p=action_prob)
        return action, action_prob

    def _optimize_model(self):
        if len(self._trajectory_buffer) != self._trajectory_buffer.maxlen:
            return

        for epoch in range(self._config.epochs):
            self._optimize_model_batch()

    def _optimize_model_batch(self):
        batch_size = self._config.batch_size
        buffer_list = list(self._trajectory_buffer)
        for i in range(0, len(buffer_list), batch_size):
            batch = buffer_list[i : i + batch_size]
            self._optimize_model_minibatch(batch)

    def _optimize_model_minibatch(self, trajectories: List[Trajectory]):
        batch = Trajectory(*zip(*trajectories))

        num_agents = len(batch.state[0].values())

        state_batch = observations_seperate_to_torch(batch.state)
        action_batch = torch.tensor(
            [torch.tensor(a) for reward in batch.action for a in reward.values()]
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            [torch.tensor(r) for reward in batch.reward for r in reward.values()]
        )
        log_probs = compute_log_probs(batch.action_probs, batch.action)

        new_actions, new_action_probs = self._get_actions(batch.state)
        new_log_probs = compute_log_probs(new_action_probs, batch.action)

        state_action_values = self._predict_policy_values(state_batch, action_batch)

        next_state_values = torch.zeros(self._config.batch_size * num_agents)
        self._predict_target_values(
            non_final_next_states, next_state_values, non_final_mask
        )
        expected_state_action_values = self._expected_state_action_values(
            next_state_values, reward_batch
        )

        loss = ppo_loss(log_probs, new_log_probs, expected_state_action_values)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
