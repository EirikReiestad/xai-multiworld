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
        self._input_dim = input_dim
        self.conv_layers = self._build_conv_layers(input_dim, conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class FCProcessor(TorchModule):
    def __init__(
        self, input_dim: np.ndarray, hidden_units: tuple[int, ...], output_dim: int
    ):
        super(FCProcessor, self).__init__()
        self._input_dim = input_dim
        self.fc_layers = self._build_fc_layers(hidden_units, input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
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
        self._fc0 = self._build_fc_layers(
            state_dim.box, hidden_units, action_dim.discrete
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), f"Input must be a type {torch.Tensor}"
        assert x.dim() != 1, f"Input must have a batch dimension, got {x.shape}"
        x = self._fc0(x)
        return x
