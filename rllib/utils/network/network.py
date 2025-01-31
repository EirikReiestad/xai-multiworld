import numpy as np
import torch
import torch.nn as nn

from rllib.utils.spaces import ActionSpace, ObservationSpace


def get_output_size(network: nn.Module, input_dim: np.ndarray) -> int:
    device = next(network.parameters()).device
    with torch.no_grad():
        dummy_input = torch.zeros(*input_dim).to(device).unsqueeze(0)
        print(dummy_input)
        output = network(dummy_input)
    return output.numel()


def observation_space_check(state_dim: ObservationSpace):
    assert state_dim.box is not None, f"State space must be continuous, got {state_dim}"
    assert (
        state_dim.discrete is not None
    ), f"State space must be discrete, got {state_dim}"


def action_space_check(action_space: ActionSpace):
    assert hasattr(
        action_space, "n"
    ), f"Action space must have attribute 'n', got {action_space}"
