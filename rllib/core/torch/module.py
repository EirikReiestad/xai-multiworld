import torch.nn as nn
import torch
from abc import abstractmethod, ABC


class TorchModule(nn.Module, ABC):
    def __init__(self):
        super(TorchModule, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
