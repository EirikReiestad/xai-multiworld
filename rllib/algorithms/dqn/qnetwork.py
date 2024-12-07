import numpy as np
import torch
import torch.nn as nn

from rllib.core.torch.module import TorchModule


class QNetwork(TorchModule):
    def __init__(
        self,
        state_dim: np.ndarray,
        action_dim: int,
        conv_layers: tuple[int, ...] = (32, 64, 64),
        hidden_units: tuple[int, ...] = (3, 32),
    ):
        super().__init__()
        self._state_dim = state_dim
        self._output_dim = action_dim
        self._hidden_units = hidden_units

        self.conv_layers = self._build_conv_layers(state_dim, conv_layers)
        conv_out_size = self._conv_layer_output_size(self._state_dim)
        self._fc_layers = self._build_fc_layers(
            self._hidden_units, conv_out_size, self._output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = self.conv_feature(x)
        x = torch.flatten(x, start_dim=1)
        x = x.to(device)
        x = self.fc_feature(x)
        return x

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
        hidden_units: tuple[int, ...],
        input_dim: int,
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
