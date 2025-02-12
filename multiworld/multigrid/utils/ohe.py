import numpy as np

from multiworld.core.constants import COLORS
from multiworld.multigrid.core.constants import Direction
from multiworld.swarm.core.constants import WorldObjectType
from multiworld.swarm.core.world_object import WorldObject

ENCODE_DIM = WorldObject.dim
N_TYPES = 5
N_COLORS = len(COLORS)
N_STATES = 4

OHE_GRID_OBJECT_DIM = N_TYPES + N_COLORS + N_STATES
OHE_GRID_OBJECT_DIM_MINIMAL = N_TYPES


def ohe_direction(direction: int) -> np.ndarray:
    result = np.array([direction == d for d in Direction], dtype=np.float32)
    return result


def ohe_agent(obj: np.ndarray, minimal: bool) -> np.ndarray:
    type_ = obj[WorldObject.TYPE]
    color = obj[WorldObject.COLOR]
    direction = obj[WorldObject.STATE]

    ohe_type = ohe_int(type_, N_TYPES)
    if minimal:
        return ohe_type
    ohe_color = ohe_int(color, N_COLORS)
    ohe_dir = ohe_direction(direction)
    return np.concatenate((ohe_type, ohe_color, ohe_dir))


def ohe_int(num: int, max: int) -> np.ndarray:
    assert (
        num <= max
    ), f"The OHE doesn't support such large numbers {num}. Maximum is {max}."

    ohe = np.zeros(max, dtype=np.int_)
    ohe[num] = 1
    return ohe


def ohe_grid_object(obj: np.ndarray, minimal: bool) -> np.ndarray:
    type_ = obj[WorldObject.TYPE]

    if type_ == WorldObjectType.agent.to_index():
        return ohe_agent(obj, minimal)

    color = obj[WorldObject.COLOR]
    state = obj[WorldObject.STATE]

    ohe_type = ohe_int(type_, N_TYPES)
    if minimal:
        return ohe_type
    ohe_color = ohe_int(color, N_COLORS)
    ohe_state = ohe_int(state, N_STATES)
    return np.concatenate((ohe_type, ohe_color, ohe_state))
