import numpy as np
import torch
import torch.nn as nn

from rllib.core.torch.module import TorchModule, build_conv_layers, build_fc_layers
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from rllib.utils.spaces import ObservationSpace, ActionSpace
from rllib.core.network.processor import ConvProcessor, FCProcessor


class MultiInputNetwork(TorchModule):
    def __init__(
        self,
        state_dim: ObservationSpace,
        action_dim: ActionSpace,
        conv_layers: tuple[int, ...] = (32, 64, 64),
        hidden_units: tuple[int, ...] = (128, 128),
    ):
        super(MultiInputNetwork, self).__init__()
        assert (
            state_dim.box is not None
        ), f"State space must be continuous, got {state_dim}"
        assert (
            state_dim.discrete is not None
        ), f"State space must be discrete, got {state_dim}"
        assert (
            action_dim.discrete is not None
        ), f"Action space must be discrete, got {action_dim}"
        self._conv0 = ConvProcessor(state_dim.box, conv_layers)
        conv_output_size = self._conv0._get_conv_layer_output_size(state_dim.box)
        self._fc0 = FCProcessor(
            int(state_dim.discrete), hidden_units, int(action_dim.discrete)
        )
        fc_output_size = hidden_units[-1]
        final_input_size = conv_output_size + int(fc_output_size)
        self._fc_final = FCProcessor(
            final_input_size, hidden_units, int(action_dim.discrete)
        )

    def forward(self, x_img: torch.Tensor, x_dir: torch.Tensor) -> torch.Tensor:
        x_img = x_img.permute(0, 3, 1, 2)
        x_img = self._conv0(x_img)
        x_img = x_img.view(x_img.size(0), -1)

        x_dir = self._fc0(x_dir)

        print(x_img.shape, x_dir.shape)

        x = torch.cat([x_img, x_dir], dim=1)
        print(x.shape)
        x = self._fc_final(x)
        return x
