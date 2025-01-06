from itertools import chain
from numpy.typing import NDArray
from typing import Dict, SupportsFloat, List
from multigrid.utils.typing import ObsType

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
    observation: ObsType,
) -> torch.Tensor:
    """
    Convert a dictionary of observations to a list of torch tensors
    """
    return torch.tensor(observation, dtype=torch.float32).unsqueeze(0)


def observations_seperate_to_torch(
    observations: List[ObsType], skip_none=False
) -> List[torch.Tensor]:
    torch_observations = [
        observation_to_torch(obs)
        for obs in observations
        if obs is not None or not skip_none
    ]
    transposed = [np.array(tensors) for tensors in zip(*torch_observations)]
    return [torch.tensor(tensor) for tensor in transposed]


def remove_none_dict(observations: Dict[str, Dict[str, NDArray]]):
    observation_copy = observations.copy()
    for key, value in observations.items():
        if value is None:
            del observation_copy[key]
    return observations


def remove_none_observations(
    observations: list[Dict[str, Dict[str, NDArray]]],
) -> List[Dict[str, Dict[str, NDArray]]]:
    """
    Remove None observations from a list of observations
    """
    return [
        remove_none_dict(observation)
        for observation in observations
        if observation is not None
    ]
