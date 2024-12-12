import numpy as np
import torch
import torch.nn as nn


def get_output_size(network: nn.Module, input_dim: np.ndarray) -> int:
    device = next(network.parameters()).device
    with torch.no_grad():
        dummy_input = torch.zeros(*input_dim).to(device).unsqueeze(0)
        output = network(dummy_input)
    return output.numel()
