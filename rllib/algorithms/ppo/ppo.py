from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.ppo.ppo_config import PPOConfig
from rllib.utils.dqn.preprocessing import preprocess_next_observations
from multigrid.utils.typing import AgentID, ObsType
from typing import SupportsFloat, Any
import torch


class PPO(Algorithm):
    pass

    def __init__(self, config: PPOConfig):
        super().__init__(config)
        self._config = config

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
        self._optimize_model()
        self._hard_update_target()

    def log_episode(self):
        super().log_episode()
        self.log_model(self._policy_net, f"model_{self._episodes_done}")
        self.add_log("eps_threshold", self._eps_threshold)

    def predict(self, observation: dict[AgentID, ObsType]) -> dict[AgentID, int]:
        sample = np.random.rand()
        self._eps_threshold = self._config.eps_end + (
            self._config.eps_start - self._config.eps_end
        ) * np.exp(-1.0 * self._steps_done / self._config.eps_decay)
        if sample > self._eps_threshold:
            actions = self._get_policy_actions(observation)
        else:
            actions = self._get_random_actions(observation)
        return actions

    def load_model(self, model: Mapping[str, Any]):
        self._policy_net.load_state_dict(model)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._policy_net.eval()
        self._target_net.eval()

    @property
    def model(self) -> nn.Module:
        return self._policy_net

    def _get_policy_actions(
        self, observations: dict[AgentID, ObsType]
    ) -> dict[AgentID, int]:
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = self._get_policy_action(obs)
        return actions

    def _get_policy_action(self, observation: ObsType) -> Action:
        with torch.no_grad():
            torch_obs = observation_to_torch_unsqueeze(observation)
            action = self._policy_net(*torch_obs).argmax().item()
        return action

    def _get_random_actions(
        self, observations: dict[AgentID, ObsType]
    ) -> dict[AgentID, int]:
        actions = {}
        for agent_id in observations.keys():
            actions[agent_id] = np.random.randint(self.action_space.discrete)
        return actions

    def _optimize_model(self):
        if len(self._memory) < self._config.batch_size:
            return

        transitions = self._memory.sample(self._config.batch_size)

        batch = Transition(*zip(*transitions))

        num_agents = len(batch.state[0].values())

        non_final_mask = get_non_final_mask(batch.next_state)
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

        state_action_values = self._predict_policy_values(state_batch, action_batch)

        next_state_values = torch.zeros(self._config.batch_size * num_agents)
        self._predict_target_values(
            non_final_next_states, next_state_values, non_final_mask
        )
        expected_state_action_values = self._expected_state_action_values(
            next_state_values, reward_batch
        )

        loss = self._compute_loss(state_action_values, expected_state_action_values)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _predict_policy_values(
        self, state: List[torch.Tensor], action_batch: torch.Tensor
    ) -> torch.Tensor:
        return self._policy_net(*state).gather(1, action_batch)

    def _predict_target_values(
        self,
        non_final_next_states: List[torch.Tensor],
        next_state_values: torch.Tensor,
        non_final_mask: List[bool],
    ) -> torch.Tensor:
        with torch.no_grad():
            output = self._target_net(*non_final_next_states).max(1).values
            next_state_values[non_final_mask] = output
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
