from collections import namedtuple
from rllib.core.memory.memory import Memory

Trajectory = namedtuple("Trajectory", ("state", "action", "action_prob", "reward"))


class TrajectoryBuffer(Memory[Trajectory]):
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer with a fixed capacity.

        :param capacity: Maximum number of transitions to store.
        """
        super().__init__(capacity, Trajectory)
