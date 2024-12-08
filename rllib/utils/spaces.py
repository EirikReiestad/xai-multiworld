from typing import NamedTuple, Tuple, Union, Optional
import gymnasium as gym
import numpy as np


class ObservationSpace(NamedTuple):
    box: Optional[np.ndarray] = None
    discrete: Optional[np.int_] = None


def build_observation_space(
    observation_space: gym.spaces.Space,
) -> list[ObservationSpace]:
    if isinstance(observation_space, gym.spaces.Box):
        return [ObservationSpace(box=observation_space.shape[0])]
    if isinstance(observation_space, gym.spaces.Discrete):
        print("Discrete")
        return [ObservationSpace(discrete=observation_space.n)]
    if isinstance(observation_space, gym.spaces.Dict):
        print("Dict")
        return [build_observation_space(space) for space in observation_space.values()]
    raise ValueError(f"Unsupported observation space: {observation_space}")


class ActionSpace(NamedTuple):
    discrete: Optional[np.int_] = None


def build_action_space(
    action_space: gym.spaces.Space,
) -> list[ActionSpace]:
    if isinstance(action_space, gym.spaces.Discrete):
        return [ActionSpace(discrete=action_space.n)]
    raise ValueError(f"Unsupported action space: {action_space}")
