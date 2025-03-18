import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils.common.numpy_collections import NumpyEncoder


class Observation(np.ndarray):
    ID = 0
    LABEL = 1
    TERMINATION = 2
    TRUNCATION = 3
    OBSERVATION = slice(4, None)

    dim = 4 + 1

    def __new__(cls, *dims: int):
        obj = np.empty(dims + (cls.dim,), dtype=object).view(cls)
        obj[..., cls.ID] = None
        obj[..., cls.LABEL] = None
        obj[..., cls.TERMINATION] = None
        obj[..., cls.TRUNCATION] = None
        obj[..., cls.OBSERVATION] = None

        return obj


def observation_from_observation_file(path: str) -> Observation:
    assert path.endswith(".json")
    data = json.load(open(path))

    observations = []
    labels = []
    terminations = []
    truncations = []
    for d in data:
        obs = list(d["observations"].values())
        observations.extend(obs)
        label = list(d["actions"].values())
        labels.extend(label)
        terms = list(d["terminations"].values())
        terminations.extend(terms)
        truncs = list(d["truncations"].values())
        truncations.extend(truncs)

    num_observations = len(observations)
    obs = Observation(num_observations)

    ids = [i for i in range(num_observations)]

    obs[..., Observation.ID] = ids
    obs[..., Observation.LABEL] = labels
    obs[..., Observation.TERMINATION] = terminations
    obs[..., Observation.TRUNCATION] = truncations
    obs[..., Observation.OBSERVATION] = np.array(observations, dtype=object).reshape(
        num_observations, 1
    )
    return obs


def observation_from_file(path: str) -> Observation:
    assert path.endswith(".json")
    json_data = json.load(open(path))
    return observation_from_dict(json_data)


def observations_from_file(path: str) -> Observation:
    assert path.endswith(".json")
    json_data = json.load(open(path))
    return observations_from_dict(json_data)


def observation_to_file(observations: Observation, path: str):
    assert path.endswith(".json")
    with open(path, "w") as f:
        json.dump(observations, f, indent=4, cls=NumpyEncoder)


def observations_from_dict(data: List[Dict]) -> Observation:
    observations = []
    labels = []
    terminations = []
    truncations = []

    for d in data:
        obs = d["observations"].values()
        label = d["actions"].values()
        terms = d["terminations"].values()
        truncs = d["truncations"].values()
        observations.extend(obs)
        labels.extend(label)
        terminations.extend(terms)
        truncations.extend(truncs)

    num_observations = len(observations)

    obs = Observation(num_observations)

    ids = [i for i in range(num_observations)]

    obs[..., Observation.ID] = ids
    obs[..., Observation.LABEL] = labels
    obs[..., Observation.TERMINATION] = terminations
    obs[..., Observation.TRUNCATION] = truncations
    obs[..., Observation.OBSERVATION] = np.array(observations, dtype=object).reshape(
        num_observations, 1
    )
    return obs


def observation_from_dict(data: List[Dict]) -> Observation:
    num_observations = len(data)

    obs = Observation(num_observations)

    ids = [i for i in range(num_observations)]
    labels = [0] * num_observations

    obs[..., Observation.ID] = ids
    obs[..., Observation.LABEL] = labels
    obs[..., Observation.TERMINATION] = False
    obs[..., Observation.TRUNCATION] = False
    obs[..., Observation.OBSERVATION] = np.array(data, dtype=object).reshape(
        num_observations, 1
    )
    return obs


def split_observation(
    observation: Observation, ratio: float, random: bool = True
) -> Tuple[Observation, Observation]:
    if random is False:
        num_observations = observation.shape[0]
        split_index = int(num_observations * ratio)
        return observation[:split_index], observation[split_index:]

    num_observations = observation.shape[0]
    indices = np.arange(num_observations)
    np.random.shuffle(indices)

    split_index = int(num_observations * ratio)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_observation = observation[train_indices]
    test_observation = observation[test_indices]

    return train_observation, test_observation


def observation_data_to_torch(observation: Observation) -> Tuple[List, List]:
    data = [
        [
            torch.tensor(v, dtype=torch.float32, requires_grad=True)
            for v in obs[0].values()
        ]
        for obs in observation[..., Observation.OBSERVATION]
    ]
    labels = observation[..., Observation.LABEL]
    return data, labels


def observation_data_to_numpy(observation: Observation) -> List:
    data = [
        [np.array(v) for v in obs[0].values()]
        for obs in observation[..., Observation.OBSERVATION]
    ]
    return data


def zipped_torch_observation_data(observation: List) -> List:
    """
    If the observation is a 2D array, this function will return a list of tuples.
    """
    return [torch.stack(tup) for tup in zip(*observation)]


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


def zip_observation_data(observation: Observation) -> Tuple[List, List]:
    assert isinstance(observation, Observation)
    data, labels = observation_data_to_torch(observation)
    return zipped_torch_observation_data(data), labels


def load_and_split_observation(
    concept: str, split_ratio=0.8, concept_path=os.path.join("assets", "concepts")
) -> Tuple[Observation, Observation]:
    observation = observation_from_file(os.path.join(concept_path, concept + ".json"))
    return split_observation(observation, split_ratio)


def randomize_observations(observation: Observation) -> Observation:
    np.random.shuffle(observation)


def normalize_observations(
    observation: Observation, a: float = 0, b: float = 1
) -> Observation:
    data = observation[..., Observation.OBSERVATION].copy()
    data = [obs[0] for obs in data]
    global_image_min = np.min(np.array([obs["observation"] for obs in data]))
    global_image_max = np.max(np.array([obs["observation"] for obs in data]))

    global_dir_min = np.min(np.array([obs["direction"] for obs in data]))
    global_dir_max = np.max(np.array([obs["direction"] for obs in data]))

    for obs in data:
        obs["observation"] = (obs["observation"] - global_image_min) / (
            global_image_max - global_image_min
        ) * (b - a) + a
        obs["direction"] = (
            obs["direction"]
            - global_dir_min / (global_dir_max - global_dir_min) * (b - a)
            + a
        )
    observation[..., Observation.OBSERVATION] = [[obs] for obs in data]
    return observation


def filter_observations(obs: Observation) -> Observation:
    mask = (obs[..., Observation.TERMINATION] == False) | (
        obs[..., Observation.TRUNCATION] == False
    )
    return obs[mask]
