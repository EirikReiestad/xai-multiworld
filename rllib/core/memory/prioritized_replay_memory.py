import random
from collections import namedtuple
from typing import List, Tuple

from rllib.core.memory.memory import Memory

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "priority")
)


class PrioritizedReplayMemory(Memory[Transition]):
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize the prioritized replay buffer with a fixed capacity.

        :param capacity: Maximum number of transitions to store.
        :param alpha: Exponent to control the prioritization, higher alpha means higher prioritization.
        """
        super().__init__(capacity, Transition)
        self.alpha = alpha
        self.priorities = [1.0] * capacity

    def add(self, **kwargs):
        """
        Add a transition to the buffer with a priority.
        """
        priority = kwargs.get("priority", 1.0)

        if self.maxlen is None or len(self) < self.maxlen:
            super().add(**kwargs)
            self.priorities[len(self) - 1] = priority
        else:
            min_priority_index = self.priorities.index(min(self.priorities))
            self[min_priority_index] = self._item_type(**kwargs)
            self.priorities[min_priority_index] = priority

    def sample(self, batch_size: int) -> Tuple[List[Transition], List[int]]:
        """
        Sample a batch of transitions based on priorities.

        :param batch_size: Number of transitions to sample.
        :return: List of sampled transitions with priority-based probability.
        """
        priorities = [t.priority for t in self]
        total_priority = sum(priorities)
        probabilities = [priority / total_priority for priority in priorities]

        indices = random.choices(range(len(self)), probabilities, k=batch_size)
        sampled_items = [self[i] for i in indices]
        return sampled_items, indices

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update the priorities of the sampled transitions.

        :param indices: List of indices for the transitions to update.
        :param priorities: List of new priorities corresponding to the transitions.
        """
        for idx, priority in zip(indices, priorities):
            self[idx] = self[idx]._replace(priority=priority)


def compute_td_errors(loss, batch_size):
    """
    Computes the TD errors based on the loss value. This function can be enhanced
    by calculating errors for each transition based on the batch and loss.
    For simplicity, we assume `loss` holds the TD errors.
    """
    return [abs(loss)] * batch_size
