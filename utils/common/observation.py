import json
from typing import List, Dict

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
