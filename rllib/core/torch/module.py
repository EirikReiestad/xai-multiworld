from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class TorchModule(nn.Module, ABC):
    def __init__(self):
        super(TorchModule, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_conv_layers(
    input_dim: np.ndarray,
    conv_layers: tuple[int, ...],
) -> nn.Sequential:
    layers = []
    channels = input_dim[-1]
    for hidden_dim in conv_layers:
        layers.append(nn.Conv2d(channels, hidden_dim, kernel_size=3, stride=1))
        layers.append(nn.ReLU())
        channels = hidden_dim
    return nn.Sequential(*layers)


def build_fc_layers(
    input_dim: int,
    hidden_units: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    layers = []
    for hidden_dim in hidden_units:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        input_dim = hidden_dim

    if output_dim is not None:
        layers.append(nn.Linear(hidden_units[-1], output_dim))
    return nn.Sequential(*layers)
