from typing import Any, Dict, Mapping, SupportsFloat

import torch.nn as nn

from multiworld.utils.typing import AgentID, ObsType
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.utils.dqn.preprocessing import preprocess_next_observations


class MDQN(Algorithm):
    def __init__(self, agents: int, config: AlgorithmConfig, dqn_config: DQNConfig):
        assert (
            dqn_config._wandb_project is None
        ), "Ups, we will only run one wandb project at a time:) So please deactivate wandb for the DQN Config and add it to the AlgorithmConfig, thank you!"
        super().__init__(config)
        self._config = config
        self._dqns = {key: DQN(dqn_config) for key in range(agents)}

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

        for key in self._dqns.keys():
            self._dqns[key]._steps_done = self._steps_done

            self._dqns[key]._memory.add_dict(
                keys=[key],
                state={key: observations[key]},
                action={key: actions[key]},
                next_state={key: next_obs[key]},
                reward={key: rewards[key]},
            )

        self._optimize_model()
        self._hard_update_target()

    def log_episode(self):
        super().log_episode()
        for key in self._dqns.keys():
            metadata = {
                "agents": len(self._env.agents),
                "width": self._env._width,
                "height": self._env._height,
                "eps_threshold": self._dqns[key]._eps_threshold,
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

        for key in self._dqns.keys():
            loss = self._dqns[key]._optimize_model()
            if loss is not None:
                losses[key] = loss

    def _hard_update_target(self):
        for key in self._dqns.keys():
            self._dqns[key]._hard_update_target()

    def _soft_update_target(self):
        for key in self._dqns.keys():
            self._dqns[key]._soft_update_target()
