import logging
from typing import Any, Dict, List, Mapping, SupportsFloat, Tuple

import torch
import torch.nn as nn

from multiworld.utils.typing import AgentID, ObsType
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.ppo.ppo_config import PPOConfig
from rllib.core.algorithms.gae import GAE
from rllib.core.memory.trajectory_buffer import Trajectory, TrajectoryBuffer
from rllib.core.network.network import Network
from rllib.core.torch.module import TorchModule
from rllib.utils.ppo.calculations import compute_log_probs, compute_returns, ppo_loss
from rllib.utils.spaces import DiscreteActionSpace
from rllib.utils.torch.processing import (
    leaf_value_to_torch,
    observations_seperate_to_torch,
    torch_stack_inner_list,
)
from utils.common.collections import zip_dict_list
from utils.core.wandb import LogMethod

"""
PPO Paper: https://arxiv.org/abs/1707.06347
"""

torch.autograd.set_detect_anomaly(True)


class PPO(Algorithm):
    _actor_net: TorchModule
    _critic_net: TorchModule

    def __init__(self, config: PPOConfig):
        super().__init__(config)
        self._config = config
        actor_network = Network(
            self._config._network_type,
            self.observation_space,
            self.action_space,
            self._config.conv_layers,
            self._config.hidden_units,
        )
        critic_network = Network(
            self._config._network_type,
            self.observation_space,
            DiscreteActionSpace(1),
            self._config.conv_layers,
            self._config.hidden_units,
        )
        self._actor_net = actor_network()
        self._critic_net = critic_network()

        # NOTE: Remove when debug is resolved
        for name, param in self._actor_net.named_parameters():
            if not param.requires_grad:
                logging.info(f"Parameter {name} requires grad: {param.requires_grad}")

        self._action_logits = {}
        self._values = {}
        self._trajectory_buffer = TrajectoryBuffer(self._config.batch_size)

        self._optimizer = torch.optim.AdamW(
            self._actor_net.parameters(), lr=config.learning_rate, amsgrad=True
        )

    def train_step(
        self,
        observations: dict[AgentID, ObsType],
        next_observations: dict[AgentID, ObsType],
        actions: dict[AgentID, int],
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
        truncations: dict[AgentID, bool],
        step: int,
        infos: dict[AgentID, dict[str, Any]],
    ):
        dones = {}
        for key in terminations.keys():
            dones[key] = terminations[key] or truncations[key]
        assert len(self._values) != 0
        assert len(self._action_logits) != 0

        self._trajectory_buffer.add(
            states=observations,
            actions=actions,
            action_probs={
                key: value.clone() for key, value in self._action_logits.copy().items()
            },
            values={key: value.clone() for key, value in self._values.copy().items()},
            rewards=rewards,
            dones=dones,
            step=step,
        )
        self._values.clear()
        self._action_logits.clear()

        self._optimize_model()

    def log_episode(self):
        super().log_episode()
        self.log_model(
            self._actor_net, f"model_actor_{self._episodes_done}", self._episodes_done
        )
        self.log_model(
            self._critic_net, f"model_critic_{self._episodes_done}", self._episodes_done
        )

    def predict(self, observation: Dict[AgentID, ObsType]) -> Dict[AgentID, int]:
        actions, action_logits, values = self._predict(observation, requires_grad=False)
        self._action_logits = {
            key: value.detach() for key, value in action_logits.items()
        }
        self._values = {key: value.clone() for key, value in values.items()}
        return actions

    def _predict(
        self, observation: Dict[AgentID, ObsType], requires_grad: bool = False
    ) -> Tuple[
        Dict[AgentID, int],
        Dict[AgentID, torch.Tensor],
        Dict[AgentID, torch.Tensor],
    ]:
        action_probabilities, policy_values = self._get_policy_values(
            list(observation.values()), requires_grad=requires_grad
        )
        actions = {}
        action_logits = {}
        values = {}
        for key, action_logit in zip(observation.keys(), action_probabilities):
            action, log_prob = self._get_action(action_logit)
            action_logits[key] = action_logit
            actions[key] = action.item()
            values[key] = policy_values[key]
        return actions, action_logits, values

    def load_model(self, models: Tuple[Mapping[str, Any], Mapping[str, Any]]):
        self._actor_net.load_state_dict(models[0])
        self._actor_net.eval()
        self._critic_net.load_state_dict(models[0])
        self._critic_net.eval()

    @property
    def model(self) -> Tuple[nn.Module, nn.Module]:
        return self._actor_net, self._critic_net

    def _get_action(
        self, action_logits: torch.Tensor, std: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._config.continuous:
            return self._get_continuous_action(action_logits=action_logits, std=std)
        else:
            return self._get_discrete_action(action_logits=action_logits)

    def _get_discrete_action(
        self, action_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def _get_continuous_action(
        self, action_logits: torch.Tensor, std: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Normal(action_logits, std)
        normal_sample = dist.sample()
        action = torch.tanh(normal_sample)
        action = (
            action * (self.action_space.high - self.action_space.low) / 2.0
            + (self.action_space.high + self.action_space.low) / 2.0
        )
        log_prob = dist.log_prob(action).sum(-1)
        return action_logits, log_prob

    def _get_policy_values(
        self, observations: List[ObsType], requires_grad: bool = False
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        torch_observations = observations_seperate_to_torch(observations)
        if requires_grad:
            return self._actor_net(*torch_observations), self._critic_net(
                *torch_observations
            )
        with torch.no_grad():
            return self._actor_net(*torch_observations), self._critic_net(
                *torch_observations
            )

    def _optimize_model(self):
        if len(self._trajectory_buffer) != self._trajectory_buffer.maxlen:
            return

        for epoch in range(self._config.epochs):
            self._optimize_model_batch()
        self._trajectory_buffer.clear()

    def _optimize_model_batch(self):
        mini_batch_size = self._config.mini_batch_size
        buffer_list = list(self._trajectory_buffer)
        self._optimize_model_minibatch(buffer_list)
        return
        for i in range(0, len(buffer_list), mini_batch_size):
            batch = buffer_list[i : i + mini_batch_size]
            self._optimize_model_minibatch(batch)

    def _optimize_model_minibatch(self, trajectories: List[Trajectory]) -> float:
        loss = 0
        buff = []
        for traj in trajectories:
            if traj.step == 0:
                if len(buff) > 1:
                    loss += self._optimize_model_minibatch_episode(buff)
                buff = []
            buff.append(traj)
        return loss
        if len(buff) > 1:
            loss += self._optimize_model_minibatch_episode(buff)
        return loss

    def _optimize_model_minibatch_episode(
        self, trajectories: List[Trajectory]
    ) -> float:
        batch = Trajectory(*zip(*trajectories))

        num_agents = len(batch.states[0].keys())
        action_logits_batch = zip_dict_list(batch.action_probs)
        state_batch = zip_dict_list(batch.states)
        action_batch = zip_dict_list(batch.actions)
        reward_batch = zip_dict_list(batch.rewards)
        value_batch = zip_dict_list(batch.values)
        dones = zip_dict_list(batch.dones)

        # NOTE: The code is a bit confusing. So if there is an error or the algorithm doesn't work. Start here.
        action_batch_torch = leaf_value_to_torch(action_batch)
        log_probs = compute_log_probs(
            torch_stack_inner_list(action_batch_torch),
            torch_stack_inner_list(action_logits_batch),
            continuous=self._config.continuous,
        )

        # TODO: This can be more efficient by passing all the states at once.
        new_action_logits = []
        new_values = []
        for state in batch.states:
            _, action_logits, value = self._predict(state, requires_grad=True)
            new_action_logits.append(action_logits)
            new_values.append(value)

        gae = GAE(
            num_agents, len(state_batch[0]), self._config.gamma, self._config.lambda_
        )

        new_value_batch = zip_dict_list(new_values)
        advantages = gae(dones, reward_batch, value_batch)
        advantages_tensor = torch_stack_inner_list(advantages).detach()
        normalized_advantages = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-10
        )

        average_advantages = float(torch.mean(torch_stack_inner_list(advantages)))
        self.add_log("advantages", average_advantages, LogMethod.AVERAGE)

        new_action_logits_batch = zip_dict_list(new_action_logits)
        new_log_probs = compute_log_probs(
            torch_stack_inner_list(action_batch_torch),
            torch_stack_inner_list(new_action_logits_batch),
            continuous=self._config.continuous,
        )
        reward_batch = torch_stack_inner_list(leaf_value_to_torch(reward_batch))
        new_value_batch = torch_stack_inner_list(new_value_batch)
        value_batch = torch_stack_inner_list(value_batch)

        returns = compute_returns(reward_batch, self._config.gamma)

        policy_loss, value_loss, entropy_loss = ppo_loss(
            log_probs,
            new_log_probs,
            normalized_advantages,
            new_value_batch.view(-1),
            returns.view(-1),
            self._config.epsilon,
        )

        self.add_log("policy_loss", policy_loss.item(), LogMethod.AVERAGE)
        self.add_log("value_loss", value_loss.item(), LogMethod.AVERAGE)
        self.add_log("entropy_loss", entropy_loss.item(), LogMethod.AVERAGE)

        loss = (
            policy_loss
            + value_loss * self._config.value_weight
            + entropy_loss * self._config.entropy_weight
        )

        self.add_log("loss", loss.item(), LogMethod.AVERAGE)

        if torch.isinf(loss) or torch.isnan(loss):
            logging.info(
                "WOOOW the loss is really high (or nan)... I think we will skip this one for now. Please check this part of the code:)"
            )
            return loss.item()

        """
        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()

        self._optimizer.zero_grad()
        value_loss.backward()
        self._optimizer.step()
        """

        self._optimizer.zero_grad()
        loss.backward()
        for name, param in self._actor_net.named_parameters():
            if param.grad is None:
                logging.info(f"{name}: {param.grad}")
        self._optimizer.step()

        return loss.item()
