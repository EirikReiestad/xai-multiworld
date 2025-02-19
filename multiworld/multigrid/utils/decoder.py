from functools import partial
from typing import Dict

import numpy as np

from multiworld.multigrid.utils.ohe import decode_ohe
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum


def decode_observation(obs: Dict, preprocessing: PreprocessingEnum) -> Dict:
    observation_grid = obs["observation"]
    decoded_obs = np.zeros((*observation_grid.shape[:2], 3))

    decode = None
    minimal = preprocessing == PreprocessingEnum.ohe_minimal
    decode = partial(decode_ohe, minimal=minimal)

    if decode is None:
        return obs

    for j, row in enumerate(observation_grid):
        for i, cell in enumerate(row):
            decoded_cell: np.ndarray = decode(cell)
            if minimal:
                decoded_cell = np.concatenate((decoded_cell, np.array([0, 0])))
            decoded_obs[j, i] = decoded_cell

    obs["observation"] = decoded_obs
    return obs
