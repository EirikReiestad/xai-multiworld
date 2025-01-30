import numpy as np
import torch
import torch.nn as nn

from rllib.core.torch.module import TorchModule
from rllib.utils.network.network import get_output_size
from rllib.utils.spaces import ObservationSpace, ActionSpace
from rllib.core.network.processor import ConvProcessor, FCProcessor
from rllib.utils.network.network import observation_space_check, action_space_check


class ActorCriticMultiInputNetwork(TorchModule):
    def __init__(
        self,
        state_dim: ObservationSpace,
        action_dim: ActionSpace,
        conv_layers: tuple[int, ...] = (32, 64, 64),
        hidden_units: tuple[int, ...] = (128, 128),
    ):
        super(ActorCriticMultiInputNetwork, self).__init__()
        observation_space_check(state_dim)
        action_space_check(action_dim)

        conv_output_size = np.prod(state_dim.box)
        self._conv0 = None
        if len(conv_layers) != 0:
            self._conv0 = ConvProcessor(state_dim.box, conv_layers)
            rolled_state_dim = np.roll(state_dim.box, shift=1)  # Channels first
            conv_output_size = get_output_size(self._conv0, rolled_state_dim)

        self._fc0 = FCProcessor(int(state_dim.discrete), hidden_units)

        rolled_state_dim = np.roll(state_dim.box, shift=1)  # Channels first
        fc_output_size = get_output_size(self._fc0, np.array([state_dim.discrete]))

        fc1_input_size = conv_output_size + fc_output_size
        self._fc1 = FCProcessor(fc1_input_size, hidden_units)

        fc1_output_size = get_output_size(self._fc1, np.array([fc1_input_size]))

        self._actor = FCProcessor(fc1_output_size, (), action_dim.n)
        self._critic = FCProcessor(fc1_output_size, (), 1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._conv0 is not None:
            x0 = x0.to(torch.float32)
            x0 = x0.permute(0, 3, 1, 2)
            x0 = self._conv0(x0)
        x0 = x0.reshape(x0.size(0), -1)

        x1 = x1.reshape(x1.size(0), -1)
        x1 = self._fc0(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self._fc1(x)
        return self._actor(x), self._critic(x)
