from itertools import chain
from typing import Dict

import gymnasium as gym
import numpy as np
import torch


def observation_to_torch(
    observation: Dict[str, gym.spaces.Space],
) -> list[torch.Tensor]:
    """
    Convert a dictionary of observations to a list of torch tensors
    """
    return [
        torch.tensor(observation[key], dtype=torch.float32)
        for key in observation.keys()
    ]


def observation_to_torch_unsqueeze(
    observation: Dict[str, gym.spaces.Space],
) -> list[torch.Tensor]:
    """
    Convert a dictionary of observations to a list of torch tensors
    """
    return [
        torch.tensor(observation[key], dtype=torch.float32).unsqueeze(0)
        for key in observation.keys()
    ]


def observations_to_torch(
    observations: Dict[str, Dict[str, gym.spaces.Space]], skip_none=False
) -> list[list[torch.Tensor]]:
    """
    Convert a dictionary of observations to a list of torch tensors
    This is useful for multi-agent environments, where observations are e.g. {"0": ..., "1": ...}
    """
    return [
        observation_to_torch(observation)
        for observation in observations.values()
        if observation is not None or not skip_none
    ]


def observations_seperate_to_torch(
    observations: list[Dict[str, Dict[str, gym.spaces.Space]]], skip_none=False
) -> list[torch.Tensor]:
    """
    Convert a dictionary of observations to a list of torch tensors
    A list of observations is expected, where each observation is a dictionary with a dictionary
    """
    obs_list = [
        observations_to_torch(observation, skip_none) for observation in observations
    ]
    obs_flattened = list(chain.from_iterable(obs_list))
    transposed = [np.array(tensors) for tensors in zip(*obs_flattened)]
    return [torch.tensor(tensor) for tensor in transposed]
