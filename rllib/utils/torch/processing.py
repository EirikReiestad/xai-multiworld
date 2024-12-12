import gymnasium as gym
from typing import Dict
import torch


def obs_to_torch(obs: Dict[str, gym.spaces.Space]) -> list[torch.Tensor]:
    """
    Convert a dictionary of observations to a list of torch tensors
    """
    return [torch.tensor(obs[key], dtype=torch.float32) for key in obs.keys()]
