from typing import List, Tuple
import torch
import torch.nn as nn

from multigrid.core.action import Action
from multigrid.core.concept import (
    get_concept_checks,
    get_concept_values,
)
from multigrid.utils.typing import ObsType
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.replay_memory import Transition
from rllib.core.concept_bottleneck.concept_bottleneck import ConceptBottleneck
from rllib.core.network.concept_bottleneck_model import ConceptBottleneckModel
from rllib.utils.dqn.misc import get_non_final_mask
from rllib.utils.torch.processing import (
    observation_to_torch_unsqueeze,
    observations_seperate_to_torch,
)


class DQNConceptBottleneckWrapper(ConceptBottleneck):
    def __init__(self, algorithm: DQN, concepts: list[str] | None = None):
        concept_checks = get_concept_checks(concepts)
        num_concepts = len(concept_checks)
        self._algorithm = algorithm

        self._algorithm._policy_net = ConceptBottleneckModel(
            self.observation_space, self.action_space, num_concepts
        )
        self._algorithm._target_net = ConceptBottleneckModel(
            self.observation_space, self.action_space, num_concepts
        )
        self._algorithm._target_net.load_state_dict(self._policy_net.state_dict())

        self._algorithm._optimize_model = self._optimize_model
        self._algorithm._predict_policy_values = self._predict_policy_values
        self._algorithm._get_policy_action = self._get_policy_action
        self._algorithm._predict_target_values = self._predict_target_values

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

        state_concept_values, state_action_values = self._predict_policy_values(
            state_batch, action_batch
        )

        next_state_values = torch.zeros(self._config.batch_size * num_agents)
        self._algorithm._predict_target_values(
            non_final_next_states, next_state_values, non_final_mask
        )
        # TODO: Remove, just a sanity check for now
        # Check that next_state_values is not all None
        assert all(next_state_values != 0), "Next state values are all None."
        expected_state_action_values = self._algorithm._expected_state_action_values(
            next_state_values, reward_batch
        )
        expected_state_concept_values = torch.Tensor(
            get_concept_values(batch.next_state)
        )

        concept_loss = self._compute_concept_loss(
            state_concept_values,
            expected_state_concept_values,
            expected_state_concept_values,
        )
        action_loss = self._algorithm._compute_action_loss(
            state_action_values, expected_state_action_values
        )

        self._optimizer.zero_grad()
        concept_loss.backward()
        self._optimizer.step()

        for param in self._policy_net.concept_head.parameters():
            param.requires_grad = False

        self._optimizer.zero_grad()
        action_loss.backward()
        self._optimizer.step()

        for param in self._policy_net.concept_head.parameters():
            param.requires_grad = True

    def _predict_policy_values(
        self, state: List[torch.Tensor], action_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        concept_pred, action_pred = self._policy_net(*state)
        return concept_pred, action_pred.gather(1, action_batch)

    def _get_policy_action(self, observation: ObsType) -> Action:
        with torch.no_grad():
            torch_obs = observation_to_torch_unsqueeze(observation)
            action, _ = self._policy_net(*torch_obs)
            action = action.argmax().item()
        return action

    def _predict_target_values(
        self,
        non_final_next_states: List[torch.Tensor],
        next_state_values: torch.Tensor,
        non_final_mask: List[bool],
    ) -> torch.Tensor:
        with torch.no_grad():
            action_output, concept_output = self._target_net(*non_final_next_states)
            action_output = action_output.max(1).values
            next_state_values[non_final_mask] = action_output
        return next_state_values

    def _compute_concept_loss(
        self,
        concept_pred: torch.Tensor,
        concept_target: torch.Tensor,  # Agents, batch size, concepts
        concept_batch: torch.Tensor,
    ) -> nn.SmoothL1Loss:
        criterion = nn.SmoothL1Loss()  # Use a suitable loss function
        criterion = criterion(concept_pred, concept_target)
        return criterion.gather(1, concept_batch)
