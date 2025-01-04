from collections import namedtuple
from rllib.core.memory.memory import Memory

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(Memory[Transition]):
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer with a fixed capacity.

        :param capacity: Maximum number of transitions to store.
        """
        super().__init__(capacity, Transition)
