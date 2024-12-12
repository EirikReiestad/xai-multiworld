import numpy as np
import torch
import torch.nn as nn

from rllib.core.torch.module import TorchModule
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from rllib.utils.spaces import ObservationSpace, ActionSpace


class ConvProcessor(TorchModule):
    def __init__(self, input_dim: np.ndarray, conv_layers: tuple[int, ...]):
        super(ConvProcessor, self).__init__()
        self.conv_layers = self._build_conv_layers(input_dim, conv_layers)

    def _get_conv_layer_output_size(self, input_dim: np.ndarray) -> int:
        device = next(self.parameters()).device
        with torch.no_grad():
            dummy_input = torch.zeros(*input_dim).to(device).unsqueeze(0)
            rearanged_input = dummy_input.permute(0, 3, 1, 2)
            output = self.conv_layers(rearanged_input)
        return output.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class FCProcessor(TorchModule):
    def __init__(self, input_dim: int, hidden_units: tuple[int, ...], output_dim: int):
        super(FCProcessor, self).__init__()
        print(input_dim)
        self.fc_layers = self._build_fc_layers(input_dim, hidden_units, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(x)


class QNetwork(TorchModule):
    def __init__(
        self,
        state_dim: ObservationSpace,
        action_dim: ActionSpace,
        conv_layers: tuple[int, ...] = (32, 64, 64),
        hidden_units: tuple[int, ...] = (128, 128),
    ):
        super(QNetwork, self).__init__()
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
        final_input_size = conv_output_size + int(state_dim.discrete)
        self._fc_final = FCProcessor(
            final_input_size, hidden_units, int(action_dim.discrete)
        )

    def forward(self, x_img: torch.Tensor, x_dir: torch.Tensor) -> torch.Tensor:
        x_img = x_img.permute(0, 3, 1, 2)
        x_img = self._conv0(x_img)
        x_dir = self._fc0(x_dir)

        x = torch.cat([x_img, x_dir], dim=1)
        x = self._fc_final(x)
        return x
