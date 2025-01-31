from typing import Any, Dict

import numpy as np

from multiworld.utils.typing import ObsType


def serialize_observation(obs_dict: ObsType):
    image_list = obs_dict["observation"].tolist()
    direction_list = obs_dict["direction"].tolist()
    serialized_dict = {"observation": image_list, "direction": direction_list}
    return serialized_dict


def deserialize_observation(data: Dict[str, Any]) -> ObsType:
    return {
        "observation": np.array(data["observation"]),
        "direction": np.array(data["direction"]),
    }
