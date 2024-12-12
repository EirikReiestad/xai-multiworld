import numpy as np
import torch
import torch.nn as nn

from rllib.core.torch.module import TorchModule, build_conv_layers, build_fc_layers
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from rllib.utils.spaces import ObservationSpace, ActionSpace


class ConvProcessor(TorchModule):
    def __init__(self, input_dim: np.ndarray, conv_layers: tuple[int, ...]):
        super(ConvProcessor, self).__init__()
        self.conv_layers = build_conv_layers(input_dim, conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class FCProcessor(TorchModule):
    def __init__(self, input_dim: int, hidden_units: tuple[int, ...], output_dim: int):
        super(FCProcessor, self).__init__()
        self.fc_layers = build_fc_layers(input_dim, hidden_units, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(x)
