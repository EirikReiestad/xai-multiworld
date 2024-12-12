from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class TorchModule(nn.Module, ABC):
    def __init__(self):
        super(TorchModule, self).__init__()

    def _get_conv_layer_output_size(self, input_dim: np.ndarray) -> int:
        device = next(self.parameters()).device
        with torch.no_grad():
            dummy_input = torch.zeros(*input_dim).to(device).unsqueeze(0)
            rearanged_input = dummy_input.permute(0, 3, 1, 2)
            output = self.conv_layers(rearanged_input)
        return output.numel()

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


def conv_layer_output_size(network: nn.Sequential, input_dim: np.ndarray) -> int:
    device = next(network.parameters()).device
    with torch.no_grad():
        dummy_input = torch.zeros(*input_dim).to(device)
        output = network.conv_feature(dummy_input)
    return output.numel()
