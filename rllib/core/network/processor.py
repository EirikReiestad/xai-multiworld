import numpy as np
import torch

from typing import Optional
from rllib.core.torch.module import TorchModule, build_conv_layers, build_fc_layers


class ConvProcessor(TorchModule):
    def __init__(self, input_dim: np.ndarray, conv_layers: tuple[int, ...]):
        super(ConvProcessor, self).__init__()
        self.conv_layers = build_conv_layers(input_dim, conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class FCProcessor(TorchModule):
    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...],
        output_dim: Optional[int] = None,
    ):
        super(FCProcessor, self).__init__()
        self.fc_layers = build_fc_layers(input_dim, hidden_units, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(x)
