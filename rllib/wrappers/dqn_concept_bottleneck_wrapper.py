from typing import Any, List, Mapping, SupportsFloat, Tuple
from numpy.typing import NDArray

import numpy as np
import torch
import torch.nn as nn

from multigrid.core.action import Action
from multigrid.core.concept import get_concept_checks
from multigrid.utils.typing import AgentID, ObsType
from multigrid.wrappers import ConceptObsWrapper
from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.replay_memory import ReplayMemory, Transition
from rllib.core.concept_bottleneck.concept_bottleneck import ConceptBottleneck
from rllib.core.network.concept_bottleneck_model import ConceptBottleneckModel
from rllib.core.network.multi_input_network import MultiInputNetwork
from rllib.utils.dqn.misc import get_non_final_mask
from rllib.utils.dqn.preprocessing import preprocess_next_observations
from rllib.utils.torch.processing import (
    observation_to_torch,
    observation_to_torch_unsqueeze,
    observations_seperate_to_torch,
    observations_to_torch,
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
        expected_state_concept_values = self._get_concept_values(batch.next_state)

        concept_loss = self._compute_concept_loss(
            state_concept_values, expected_state_concept_values, action_batch
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

    def _get_concept_values(self, next_state: List[NDArray]):
        """
        This function processes the next state observations and checks for the presence of each concept.
        It returns a bit map (list of 0 or 1) for each concept, indicating whether it's present in the given state.

        Args:
            next_state (List[NDArray]): List of state observations for multiple agents.

        Returns:
            List[NDArray]: List of concept bit maps, each representing the presence (1) or absence (0) of concepts.
        """
        # List to store the bit maps for each concept in the batch
        concept_bit_maps = []

        # Iterate over the states for each agent
        for state in next_state:
            # Initialize a list to store concept values (0 or 1) for the current state
            concept_values = []

            # For each concept, check its presence (using concept checks, as defined in `get_concept_checks`)
            for concept_check in self._concept_checks:
                # Apply the concept check to the state (assuming it returns True/False)
                concept_present = concept_check(
                    state
                )  # Will return True if concept is present, else False
                concept_values.append(1 if concept_present else 0)

            # Convert the list of 0s and 1s to a numpy array for the current state
            concept_bit_maps.append(np.array(concept_values))

        # Return the list of concept bit maps
        return concept_bit_maps

    def _compute_concept_loss(
        self,
        concept_pred: torch.Tensor,
        concept_target: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> nn.SmoothL1Loss:
        criterion = nn.SmoothL1Loss()  # Use a suitable loss function
        return criterion(concept_pred, concept_target).gather(1, action_batch)
