import json
import torch
from typing import List, Dict, Tuple

import numpy as np


class Observation(np.ndarray):
    ID = 0
    LABEL = 1
    DATA = slice(2, None)

    dim = 2 + 1

    def __new__(cls, *dims: int):
        obj = np.empty(dims + (cls.dim,), dtype=object).view(cls)
        obj[..., cls.ID] = None
        obj[..., cls.LABEL] = None
        obj[..., cls.DATA] = None

        return obj


def observation_from_file(path: str) -> Observation:
    assert path.endswith(".json")
    json_data = json.load(open(path))
    return observation_from_dict(json_data)


def observation_from_dict(data: List[Dict]) -> Observation:
    num_observations = len(data)

    obs = Observation(num_observations)

    ids = [i for i in range(num_observations)]
    labels = [0] * num_observations

    obs[..., Observation.ID] = ids
    obs[..., Observation.LABEL] = labels
    obs[..., Observation.DATA] = np.array(data, dtype=object).reshape(
        num_observations, 1
    )
    return obs


def split_observation(
    observation: Observation, ratio: float
) -> Tuple[Observation, Observation]:
    num_observations = observation.shape[0]
    split_index = int(num_observations * ratio)
    return observation[:split_index], observation[split_index:]


def observation_data_to_torch(observation: Observation) -> List:
    data = [
        [torch.tensor(v) for v in obs[0].values()]
        for obs in observation[..., Observation.DATA]
    ]
    return data


def zipped_torch_observation_data(observation: List) -> List:
    """
    If the observation is a 2D array, this function will return a list of tuples.
    """
    zipped = list(zip(*observation))
    print(len(zipped))
    return zipped
