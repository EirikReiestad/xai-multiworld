import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch


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

    @property
    def features(self):
        return self[..., self.DATA]


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
    return [torch.tensor(np.array(tup)) for tup in zip(*observation)]


def set_require_grad(observation: List):
    for i, obs in enumerate(observation):  # Use enumerate to modify the list in place
        if isinstance(obs, List):
            set_require_grad(obs)
            continue
        elif isinstance(obs, torch.Tensor):
            observation[i] = obs.float()  # Modify the tensor in the original list
            observation[i].requires_grad = True  # Set requires_grad in place
        else:
            raise ValueError(f"Expected torch.Tensor or List, got {type(obs)}")

    assert all([obs.requires_grad for obs in observation])


def zip_observation_data(observation: Observation) -> List:
    assert isinstance(observation, Observation)
    data = observation_data_to_torch(observation)
    return zipped_torch_observation_data(data)


def load_and_split_observation(
    concept: str, split_ratio=0.8, concept_path=os.path.join("assets", "concepts")
) -> Tuple[Observation, Observation]:
    observation = observation_from_file(os.path.join(concept_path, concept + ".json"))
    return split_observation(observation, split_ratio)
