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

    def _build_conv_layers(
        self,
        input_dim: np.ndarray,
        conv_layers: tuple[int, ...],
    ) -> nn.Sequential:
        input_size = input_dim.shape[0]
        layers = []
        for hidden_dim in conv_layers:
            layers.append(nn.Conv2d(input_size, hidden_dim, kernel_size=3, stride=1))
            layers.append(nn.ReLU())
            input_size = hidden_dim
        return nn.Sequential(*layers)

    def _build_fc_layers(
        self,
        input_dim: np.ndarray,
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

    def _conv_layer_output_size(self, input_dim: np.ndarray) -> int:
        device = next(self.parameters()).device
        with torch.no_grad():
            dummy_input = torch.zeros(*input_dim).to(device)
            output = self.conv_feature(dummy_input)
        return output.numel()
