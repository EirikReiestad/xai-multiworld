import numpy as np
import torch

from rllib.core.torch.module import TorchModule
from rllib.utils.network.network import get_output_size
from rllib.utils.spaces import ObservationSpace, ActionSpace
from rllib.core.network.processor import ConvProcessor, FCProcessor
from rllib.utils.network.network import observation_space_check, action_space_check


class ActorCriticNetwork(TorchModule):
    def __init__(
        self,
        state_dim: ObservationSpace,
        action_dim: ActionSpace,
        conv_layers: tuple[int, ...] = (32, 64, 64),
        hidden_units: tuple[int, ...] = (128, 128),
    ):
        super(ActorCriticNetwork, self).__init__()
        observation_space_check(state_dim)
        action_space_check(action_dim)

        conv_output_size = np.prod(state_dim.box)
        self._conv0 = None
        if len(conv_layers) != 0:
            self._conv0 = ConvProcessor(state_dim.box, conv_layers)
            rolled_state_dim = np.roll(state_dim.box, shift=1)  # Channels first
            conv_output_size = get_output_size(self._conv0, rolled_state_dim)

        self._fc0 = FCProcessor(conv_output_size, hidden_units)

        fc0_output_size = get_output_size(self._fc0, np.array([conv_output_size]))

        self._actor = FCProcessor(fc0_output_size, (), action_dim.n)
        self._critic = FCProcessor(fc0_output_size, (), 1)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        if self._conv0 is not None:
            x = x.to(torch.float32)
            x = x.permute(0, 3, 1, 2)
            x = self._conv0(x)

        x = x.reshape(x.size(0), -1)
        x = self._fc0(x)
        return self._actor(x), self._critic(x)
