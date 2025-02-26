from typing import Tuple, final

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


class MultiInputNetwork(TorchModule):
    def __init__(
        self,
        state_dim: ObservationSpace,
        action_dim: ActionSpace,
        conv_layers: Tuple[int, ...] = (32, 64, 64),
        hidden_units: Tuple[int, ...] = (128, 128),
    ):
        super(MultiInputNetwork, self).__init__()
        observation_space_check(state_dim)
        action_space_check(action_dim)

        conv_output_size = np.prod(state_dim.box)
        self._conv0 = None
        if len(conv_layers) != 0:
            self._conv0 = ConvProcessor(state_dim.box, conv_layers)
            rolled_state_dim = np.roll(state_dim.box, shift=1)  # Channels first
            conv_output_size = get_output_size(self._conv0, rolled_state_dim)

        self._fc0 = FCProcessor(
            int(state_dim.discrete), hidden_units, int(action_dim.n)
        )
        fc_output_size = get_output_size(self._fc0, np.array([state_dim.discrete]))
        final_input_size = conv_output_size + fc_output_size
        self._create_fc_final(final_input_size, hidden_units, int(action_dim.n))

    def _create_fc_final(
        self, final_input_size: int, hidden_units: tuple[int, ...], action_dim: int
    ):
        self._fc_final = FCProcessor(final_input_size, hidden_units, action_dim)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        x = x0.clone()
        y = x1.clone()

        if self._conv0 is not None:
            x = x.float()
            x = x.permute(0, 3, 1, 2)
            x = self._conv0(x)
        x = x.reshape(x.size(0), -1)

        y = y.reshape(y.size(0), -1)
        y = self._fc0(y)

        c = torch.cat([x, y], dim=1)
        f = self._fc_final(c)
        return f
