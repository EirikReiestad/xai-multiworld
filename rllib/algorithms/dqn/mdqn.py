from collections import deque
from typing import Any, Dict, Mapping, SupportsFloat

import numpy as np
import torch.nn as nn

from multiworld.utils.typing import AgentID, ObsType
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.utils.dqn.preprocessing import preprocess_next_observations
from utils.core.wandb import LogMethod


class MDQN(Algorithm):
    def __init__(
        self,
        agents: int,
        config: AlgorithmConfig,
        dqn_config: DQNConfig,
        multi_training: bool = False,
        performance_measure_len: int = 10,
    ):
        assert (
            dqn_config._wandb_project is None
        ), "Ups, we will only run one wandb project at a time:) So please deactivate wandb for the DQN Config and add it to the AlgorithmConfig, thank you!"
        super().__init__(config)
        performance_measure_len *= self._env._max_steps
        self._multi_training = multi_training
        self._config = config
        self._dqn_config = dqn_config
        self._dqns = {key: DQN(dqn_config) for key in range(agents)}
        self._performance_measure_len = performance_measure_len
        self._performance_measure = {
            key: deque(maxlen=performance_measure_len) for key in range(agents)
        }

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
        next_obs = preprocess_next_observations(
            next_observations, terminations, truncations
        )

        # Just add reward for active step, not when waiting for the others to finish
        for key, reward in rewards.items():
            if (terminations[key] or truncations[key]) and reward == 0:
                continue
            self._performance_measure[key].append(reward)

        for key in self._dqns.keys():
            self._dqns[key]._steps_done = self._steps_done

            priority = {key: 1.0 for key in next_obs.keys()}

            self._dqns[key]._memory.add_dict(
                keys=[key],
                state={key: observations[key]},
                action={key: actions[key]},
                next_state={key: next_obs[key]},
                reward={key: rewards[key]},
                priority=priority,
            )

        self._optimize_model()
        self._hard_update_target()
        self._soft_update_outlier_agents()

    def log_episode(self):
        super().log_episode()
        if self._multi_training:
            for key in self._dqns.keys():
                metadata = {
                    "agents": len(self._env.agents),
                    "width": self._env._width,
                    "height": self._env._height,
                    "preprocessing": self._env._preprocessing,
                    "network_type": self._dqn_config._network_type,
                    "eps_threshold": self._dqns[key]._eps_threshold,
                    "learning_rate": self._dqns[key]._config.learning_rate,
                    "conv_layers": self._config.conv_layers,
                    "hidden_units": self._config.hidden_units,
                }
                self.log_model(
                    self._dqns[key].model,
                    f"model_{key}_{self._episodes_done}",
                    self._episodes_done,
                    metadata,
                )
                self.add_log(f"eps_threshold_{key}", self._dqns[key]._eps_threshold)
        else:
            key = list(self._dqns.keys())[0]
            metadata = {
                "agents": len(self._env.agents),
                "width": self._env._width,
                "height": self._env._height,
                "preprocessing": self._env._preprocessing,
                "network_type": self._dqn_config._network_type,
                "eps_threshold": self._dqns[key]._eps_threshold,
                "learning_rate": self._dqns[key]._config.learning_rate,
                "conv_layers": self._config.conv_layers,
                "hidden_units": self._config.hidden_units,
            }
            self.log_model(
                self._dqns[key].model,
                f"model_{key}_{self._episodes_done}",
                self._episodes_done,
                metadata,
            )
            self.add_log(f"eps_threshold_{key}", self._dqns[key]._eps_threshold)

    def predict(self, observation: Dict[AgentID, ObsType]) -> Dict[AgentID, int]:
        actions = {}
        for key in self._dqns.keys():
            action = self._dqns[key].predict({key: observation[key]})
            for key, value in action.items():
                if actions.get(key) is not None:
                    raise ValueError(f"The action for agent {key} is returned already.")
                actions[key] = value
        return actions

    def load_model(self, model: Mapping[str, Any]):
        for key in self._dqns.keys():
            self._dqns[key].load_model(model[key])

    @property
    def model(self, key: int) -> nn.Module:
        return self._dqns[key].model

    def _optimize_model(self):
        losses = {}

        if self._multi_training:
            for key in self._dqns.keys():
                loss = self._dqns[key]._optimize_model()
                if loss is not None:
                    losses[key] = loss
        else:
            key = list(self._dqns.keys())[0]
            loss = self._dqns[key]._optimize_model()
            if loss is not None:
                losses[key] = loss

    def _soft_update_outlier_agents(self):
        if self._steps_done < self._performance_measure_len:
            return

        performance = {
            key: float(np.mean(values))
            for key, values in self._performance_measure.items()
        }
        performance_values = list(performance.values())
        median_performance = np.median(performance_values)
        q25, q75 = np.percentile(performance_values, [25, 75])
        iqr = q75 - q25

        lower_bound = median_performance - 1 * iqr

        self.add_log("outlier_update_lower_bound", lower_bound)

        sorted_performance = {
            k: v for k, v in sorted(performance.items(), key=lambda item: item[1])
        }
        best_agent = list(sorted_performance.keys())[-1]
        best_agent_network = self._dqns[best_agent].model

        extern_updates = 0
        for key, value in sorted_performance.items():
            if value < lower_bound:
                self._dqns[key]._soft_update_target(best_agent_network)
                self.add_log(f"extern_update_{key}", 1, LogMethod.CUMULATIVE)
                extern_updates += 1
                continue
            self.add_log(f"extern_update_{key}", 0, LogMethod.CUMULATIVE)
        self.add_log("extern_update", extern_updates, LogMethod.CUMULATIVE)

    def _hard_update_target(self):
        if self._multi_training:
            for key in self._dqns.keys():
                self._dqns[key]._hard_update_target()
        else:
            key = list(self._dqns.keys())[0]
            self._dqns[key]._hard_update_target()
            model = self._dqns[key].model

            for other_key in self._dqns.keys():
                if key == other_key:
                    continue
                if np.random.rand() > 0.2:  # Only updating some of the models maybe
                    continue
                self._dqns[other_key]._hard_update_target(model)

    def _soft_update_target(self):
        for key in self._dqns.keys():
            self._dqns[key]._soft_update_target()
