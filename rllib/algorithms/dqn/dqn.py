import logging
from typing import Any, Dict, List, Mapping, SupportsFloat

import numpy as np
import torch
import torch.nn as nn

from multiworld.utils.advanced_typing import Action
from multiworld.utils.typing import AgentID, ObsType
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.memory.prioritized_replay_memory import (
    PrioritizedReplayMemory,
    Transition,
    compute_td_errors,
)
from rllib.core.network.network import Network
from rllib.core.torch.module import TorchModule
from rllib.utils.dqn.misc import get_non_final_mask
from rllib.utils.dqn.preprocessing import preprocess_next_observations
from rllib.utils.torch.processing import (
    observation_to_torch_unsqueeze,
    observations_seperate_to_torch,
)
from utils.core.model_loader import ModelLoader


class DQN(Algorithm):
    _policy_net: TorchModule
    _target_net: TorchModule

    def __init__(self, config: DQNConfig):
        super().__init__(config)
        self._config = config
        # self._memory = ReplayMemory(config.replay_buffer_size)
        self._memory = PrioritizedReplayMemory(config.replay_buffer_size)
        network = Network(
            self._config._network_type,
            self.observation_space,
            self.action_space,
            self._config.conv_layers,
            self._config.hidden_units,
        )
        self._policy_net = network()
        self._target_net = network()

        if self._config._eval:
            self._policy_net.eval()
            self._target_net.eval()

        if self._config._model_path is not None:
            ModelLoader.load_model_from_path(self._config._model_path, self._policy_net)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._optimizer = torch.optim.AdamW(
            self._policy_net.parameters(), lr=config.learning_rate, amsgrad=True
        )
        self._scheduler = self._gen_scheduler()
        self._eps_threshold = np.inf

    def train_step(
        self,
        observations: Dict[AgentID, ObsType],
        next_observations: Dict[AgentID, ObsType],
        actions: Dict[AgentID, int],
        rewards: Dict[AgentID, SupportsFloat],
        terminations: Dict[AgentID, bool],
        truncations: Dict[AgentID, bool],
        step: int,
        infos: Dict[AgentID, Dict[str, Any]],
    ):
        if self._config._training is False:
            return
        next_obs = preprocess_next_observations(
            next_observations, terminations, truncations
        )

        priority = {key: 1.0 for key in next_obs.keys()}

        self._memory.add_dict(
            keys=observations.keys(),
            action=actions,
            state=observations,
            next_state=next_obs,
            reward=rewards,
            priority=priority,
        )

        if self._config.update_method == "soft":
            self._soft_update_target()
        else:
            self._hard_update_target()
        self._optimize_model()

    def log_episode(self):
        super().log_episode()
        metadata = {}
        learning_rate = (
            self._scheduler.get_last_lr()
            if self._scheduler is not None
            else self._config.learning_rate,
        )
        if isinstance(learning_rate, tuple):
            learning_rate = learning_rate[0]
        if isinstance(learning_rate, list):
            learning_rate = learning_rate[0]

        try:
            metadata = {
                "agents": len(self._env.agents),
                "width": self._env._width,
                "height": self._env._height,
                "eps_threshold": self._eps_threshold,
                "learning_rate": learning_rate,
                "conv_layers": self._config.conv_layers,
                "hidden_units": self._config.hidden_units,
            }
        except AttributeError:
            pass

        self.log_model(
            self._policy_net,
            f"model_{self._episodes_done}",
            self._episodes_done,
            metadata,
        )
        self.add_log("eps_threshold", self._eps_threshold)
        self.add_log("learning_rate", learning_rate)

    def predict(self, observation: dict[AgentID, ObsType]) -> dict[AgentID, int]:
        self._eps_threshold = self._config.eps_end + (
            self._config.eps_start - self._config.eps_end
        ) * np.exp(-1.0 * self._steps_done / self._config.eps_decay)

        actions = self._get_policy_actions(observation)
        for key, _ in actions.items():
            if np.random.rand() < self._eps_threshold:
                actions[key] = self._get_random_action()
        return actions

    def load_model(self, model: Mapping[str, Any]):
        self._policy_net.load_state_dict(model)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._policy_net.eval()
        self._target_net.eval()

    @property
    def model(self) -> nn.Module:
        return self._policy_net

    def _gen_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self._config._lr_scheduler is None:
            return
        if self._config._lr_scheduler == "cyclic":
            return torch.optim.lr_scheduler.CyclicLR(
                self._optimizer,
                base_lr=self._config._base_lr,
                max_lr=self._config._max_lr,
            )
        if self._config._lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self._optimizer,
                step_size=self._config._scheduler_step_size,
                gamma=self._config._scheduler_gamma,
            )

    def _get_policy_actions(
        self, observations: Dict[AgentID, ObsType]
    ) -> Dict[AgentID, int]:
        torch_observations = observations_seperate_to_torch(list(observations.values()))
        with torch.no_grad():
            pred_actions = self._policy_net(*torch_observations)

        actions = {}
        for key, action in zip(observations.keys(), pred_actions):
            actions[key] = action.argmax().item()
        return actions

    def _get_policy_action(self, observation: ObsType) -> Action:
        with torch.no_grad():
            torch_obs = observation_to_torch_unsqueeze(observation)
            return self._policy_net(*torch_obs).argmax().item()

    def _get_random_action(self):
        return np.random.randint(self.action_space.n)

    def _optimize_model(self) -> float | None:
        if len(self._memory) < self._config.batch_size:
            return None

        transitions, indices = self._memory.sample(self._config.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = get_non_final_mask(batch.next_state)
        non_final_next_states = observations_seperate_to_torch(
            batch.next_state, skip_none=True
        )
        if len(non_final_next_states) == 0:
            logging.warning("No non final next states, consider increasing batch size.")
            return None

        state_batch = observations_seperate_to_torch(batch.state)
        action_batch = torch.tensor(batch.action).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward)

        state_action_values = self._predict_policy_values(state_batch, action_batch)

        next_state_actions = (
            self._policy_net(*non_final_next_states).argmax(1).unsqueeze(1)
        )
        next_state_values = torch.zeros(self._config.batch_size)
        self._predict_target_values(
            non_final_next_states, next_state_values, non_final_mask, next_state_actions
        )
        expected_state_action_values = self._expected_state_action_values(
            next_state_values, reward_batch
        )

        loss = self._compute_loss(state_action_values, expected_state_action_values)

        self.add_log("loss", loss.item())

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        if self._scheduler is not None:
            self._scheduler.step()

        if loss is not None:
            td_errors = compute_td_errors(loss, self._config.batch_size)
            self._memory.update_priorities(indices=indices, priorities=td_errors)

        return loss.item()

    def _predict_policy_values(
        self, state: List[torch.Tensor], action_batch: torch.Tensor
    ) -> torch.Tensor:
        return self._policy_net(*state).gather(1, action_batch)

    def _predict_target_values(
        self,
        non_final_next_states: List[torch.Tensor],
        next_state_values: torch.Tensor,
        non_final_mask: List[bool],
        next_state_actions: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            output = self._target_net(*non_final_next_states).gather(
                1, next_state_actions
            )
            next_state_values[non_final_mask] = output.squeeze()
        return next_state_values

    def _expected_state_action_values(
        self, next_state_values: torch.Tensor, reward_batch: torch.Tensor
    ) -> torch.Tensor:
        return next_state_values * self._config.gamma + reward_batch

    def _compute_loss(
        self,
        state_action_values: torch.Tensor,
        expected_state_action_values: torch.Tensor,
    ) -> torch.nn.SmoothL1Loss:
        return self._compute_action_loss(
            state_action_values, expected_state_action_values
        )

    def _compute_action_loss(
        self,
        state_action_values: torch.Tensor,
        expected_state_action_values: torch.Tensor,
    ) -> torch.nn.SmoothL1Loss:
        criterion = torch.nn.SmoothL1Loss()
        return criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        ).mean()

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
