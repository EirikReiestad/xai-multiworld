from typing import Tuple

import numpy as np
import torch

from rllib.core.network.processor import ConvProcessor, FCProcessor
from rllib.core.torch.module import TorchModule
from rllib.utils.network.network import (
    action_space_check,
    get_output_size,
    observation_space_check,
)
from rllib.utils.spaces import ActionSpace, ObservationSpace


class ConceptBottleneckModel(TorchModule):
    def __init__(
        self,
        state_dim: ObservationSpace,
        action_dim: ActionSpace,
        concept_dim: int,
        conv_layers: tuple[int, ...] = (32, 64, 64),
        hidden_units: tuple[int, ...] = (128, 128),
        concept_hidden_units: tuple[int, ...] = (32, 64),
    ):
        super(ConceptBottleneckModel, self).__init__()
        observation_space_check(state_dim)
        action_space_check(action_dim)
        self._conv0 = ConvProcessor(state_dim.box, conv_layers)
        rolled_state_dim = np.roll(state_dim.box, shift=1)  # Channels first
        conv_output_size = get_output_size(self._conv0, rolled_state_dim)
        self._fc0 = FCProcessor(
            int(state_dim.discrete), hidden_units, int(action_dim.discrete)
        )
        fc_output_size = get_output_size(self._fc0, np.array([state_dim.discrete]))
        final_input_size = conv_output_size + fc_output_size

        self._fc_concept = FCProcessor(final_input_size, hidden_units, concept_dim)

        self._fc_final = FCProcessor(
            final_input_size, concept_hidden_units, int(action_dim.discrete)
        )

    def forward(
        self, x_img: torch.Tensor, x_dir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_img = x_img.float()
        x_img = x_img.permute(0, 3, 1, 2)
        x_img = self._conv0(x_img)
        x_img = x_img.view(x_img.size(0), -1)

        x_dir = self._fc0(x_dir)

        x = torch.cat([x_img, x_dir], dim=1)

        concept_pred = self._fc_concept(x)
        action_pred = self._fc_final(x)
        return concept_pred, action_pred
