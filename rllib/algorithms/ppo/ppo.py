from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.ppo.ppo_config import PPOConfig
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
    observations_seperate_to_torch,
)
import torch.nn as nn
from rllib.core.memory.trajectory_buffer import TrajectoryBuffer, Trajectory
from rllib.utils.ppo.calculations import compute_log_probs, ppo_loss
from rllib.core.algorithms.gae import GAE
from utils.common.numpy_collections import dict_list_to_ndarray

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
        self._values = {}
        self._trajectory_buffer = TrajectoryBuffer(self._config.batch_size)

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
        dones = {}
        for key in terminations.keys():
            dones[key] = terminations[key] or truncations[key]
        self._trajectory_buffer.add(
            states=observations,
            actions=actions,
            action_probs=self._action_probs,
            values=self._values,
            rewards=rewards,
            dones=dones,
        )
        self._optimize_model()

    def log_episode(self):
        super().log_episode()
        self.log_model(
            self._policy_net, f"model_{self._episodes_done}", self._episodes_done
        )

    def predict(self, observation: Dict[AgentID, ObsType]) -> Dict[AgentID, int]:
        actions, action_probs, values = self._predict(observation)
        self._action_probs = action_probs
        self._values = values
        return actions

    def _predict(
        self, observation: Dict[AgentID, ObsType]
    ) -> Tuple[
        Dict[AgentID, int],
        Dict[AgentID, NDArray],
        Dict[AgentID, float],
    ]:  # NOTE: A but ugly code maybe...? as there already is a def predict(...) method
        action_probabilities, policy_values = self._get_policy_values(
            list(observation.values())
        )
        actions = {}
        action_probs = {}
        values = {}
        for key, action_prob in zip(observation.keys(), action_probabilities):
            action_probs[key] = action_prob.squeeze().detach().numpy()
            actions[key] = self._get_action(action_probs[key])
            values[key] = policy_values[key].squeeze().detach().numpy()
        return actions, action_probs, values

    def load_model(self, model: Mapping[str, Any]):
        self._policy_net.load_state_dict(model)
        self._policy_net.eval()

    @property
    def model(self) -> nn.Module:
        return self._policy_net

    def _get_action(self, action_prob: NDArray) -> Action:
        action = np.random.choice(range(self.action_space.discrete), p=action_prob)
        return action

    def _get_policy_values(
        self, observations: List[ObsType]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        torch_observations = observations_seperate_to_torch(observations)
        with torch.no_grad():
            return self._policy_net(*torch_observations)

    def _optimize_model(self):
        if len(self._trajectory_buffer) != self._trajectory_buffer.maxlen:
            return

        for epoch in range(self._config.epochs):
            self._optimize_model_batch()

    def _optimize_model_batch(self):
        mini_batch_size = self._config.mini_batch_size
        buffer_list = list(self._trajectory_buffer)
        for i in range(0, len(buffer_list), mini_batch_size):
            batch = buffer_list[i : i + mini_batch_size]
            self._optimize_model_minibatch(batch)
        self._trajectory_buffer.clear()

    def _optimize_model_minibatch(self, trajectories: List[Trajectory]):
        batch = Trajectory(*zip(*trajectories))

        num_agents = len(batch.states[0].keys())

        action_prob_batch = dict_list_to_ndarray(batch.action_probs)
        state_batch = dict_list_to_ndarray(batch.states)
        action_batch = dict_list_to_ndarray(batch.actions)
        reward_batch = dict_list_to_ndarray(batch.rewards)
        value_batch = dict_list_to_ndarray(batch.values)
        dones = dict_list_to_ndarray(batch.dones)

        assert (
            (len(batch.states[0]), len(batch.states))
            == state_batch.shape[:2]
            == action_batch.shape[:2]
            == reward_batch.shape[:2]
            == action_prob_batch.shape[:2]
            == value_batch.shape[:2]
        ), f"All batch attributes should have the same shape. Should get {len(batch.states)}, got: {state_batch.shape}, {action_batch.shape}, {reward_batch.shape}, {action_prob_batch.shape}, {value_batch.shape}"

        # NOTE: The code is a bit confusing. So if there is an error or the algorithm doesn't work. Start here.
        log_probs = compute_log_probs(action_batch, action_prob_batch)

        gae = GAE(
            num_agents, len(state_batch[0]), self._config.gamma, self._config.lambda_
        )
        advantages = gae(dones, reward_batch, value_batch)

        # TODO: This can be more efficient by passing all the states at once.
        new_action_probs = []
        for state in batch.states:
            _, action_probs, _ = self._predict(state)
            new_action_probs.append(action_probs)
        new_action_probs_batch = dict_list_to_ndarray(new_action_probs)
        new_log_probs = compute_log_probs(action_batch, new_action_probs_batch)

        old_log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32)
        new_log_probs_tensor = torch.tensor(new_log_probs, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

        loss = ppo_loss(
            old_log_probs_tensor,
            new_log_probs_tensor,
            advantages_tensor,
            self._config.epsilon,
        )

        self.add_log("loss", loss.item())

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
