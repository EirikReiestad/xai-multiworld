import enum

from rllib.core.network.actor_critic_multi_input_network import (
    ActorCriticMultiInputNetwork,
)
from rllib.core.network.actor_critic_network import ActorCriticNetwork
from rllib.core.network.feed_forward_network import FeedForwardNetwork
from rllib.core.network.multi_input_network import MultiInputNetwork
from rllib.core.torch.module import TorchModule
from rllib.utils.spaces import ActionSpace, ObservationSpace


class NetworkType(enum.Enum):
    ACTORCRITIC = "actorcritic"
    MULTI_ACTORCRITIC = "multi_actorcritic"
    MULTI_INPUT = "multi_input"
    FEED_FORWARD = "feed_forward"


class Network:
    T: NetworkType
    state_dim: ObservationSpace
    action_dim: ActionSpace
    conv_layers: tuple[int, ...]
    hidden_units: tuple[int, ...]

    def __init__(
        self,
        T: NetworkType | None,
        state_dim: ObservationSpace,
        action_dim: ActionSpace,
        conv_layers: tuple[int, ...],
        hidden_units: tuple[int, ...],
    ) -> None:
        if T is None:
            T = NetworkType.FEED_FORWARD
        self.T = T
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.conv_layers = conv_layers
        self.hidden_units = hidden_units

    def __call__(self) -> TorchModule:
        if self.T == NetworkType.ACTORCRITIC:
            return ActorCriticNetwork(
                self.state_dim, self.action_dim, self.conv_layers, self.hidden_units
            )
        if self.T == NetworkType.MULTI_ACTORCRITIC:
            return ActorCriticMultiInputNetwork(
                self.state_dim, self.action_dim, self.conv_layers, self.hidden_units
            )
        if self.T == NetworkType.MULTI_INPUT:
            return MultiInputNetwork(
                self.state_dim, self.action_dim, self.conv_layers, self.hidden_units
            )
        if self.T == NetworkType.FEED_FORWARD:
            return FeedForwardNetwork(
                self.state_dim, self.action_dim, self.conv_layers, self.hidden_units
            )

        raise ValueError(f"Network type not supported: Err trying to parse {self.T}")
