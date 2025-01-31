from typing import Optional
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray


@dataclass
class ObservationSpace:
    box: Optional[np.ndarray] = None
    discrete: Optional[np.int_] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObservationSpace):
            return False

        return (
            self.box is None
            and other.box is None
            or (
                self.box is not None
                and other.box is not None
                and np.array_equal(self.box, other.box)
            )
        ) and (self.discrete == other.discrete)


def build_observation_space(
    observation_space: gym.spaces.Space,
    current_observation_space: Optional[ObservationSpace] = None,
) -> ObservationSpace:
    """
    Recursively build the observation space from the gym observation space with numpy arrays at the leaves.
    """
    if current_observation_space is None:
        current_observation_space = ObservationSpace()

    if isinstance(observation_space, gym.spaces.Box):
        assert current_observation_space.box is None, "Multiple boxes are not supported"
        current_observation_space.box = np.array(observation_space.shape)
    elif isinstance(observation_space, gym.spaces.Discrete):
        assert (
            current_observation_space.discrete is None
        ), "Multiple discrete spaces are not supported"
        current_observation_space.discrete = observation_space.n
    elif isinstance(observation_space, gym.spaces.Dict):
        for _, space in observation_space.items():
            build_observation_space(space, current_observation_space)
    else:
        raise ValueError(f"Unsupported observation space: {observation_space}")
    return current_observation_space


class ActionSpace(ABC):
    low: float
    high: float
    n: int | float

    @abstractmethod
    def sample(self) -> int | NDArray[np.float64]:
        pass

    @abstractmethod
    def contains(self, x: int | float) -> bool:
        pass


@dataclass
class DiscreteActionSpace(ActionSpace):
    n: int

    def sample(self) -> int:
        return np.random.randint(self.n)

    def contains(self, x: int) -> bool:
        return 0 <= x < self.n


@dataclass
class BoxActionSpace(ActionSpace):
    low: float
    high: float
    n: float

    def sample(self) -> int:
        return np.random.randint(low=self.low, high=self.high)

    def contains(self, x: float) -> bool:
        return self.low <= x < self.high


def build_action_space(
    action_space: gym.spaces.Space,
) -> ActionSpace:
    if isinstance(action_space, gym.spaces.Discrete):
        return DiscreteActionSpace(n=action_space.n)
    if isinstance(action_space, gym.spaces.Box):
        return BoxActionSpace(
            low=action_space.low, high=action_space.high, n=action_space.shape[0]
        )
    raise ValueError(f"Unsupported action space: {action_space}")
