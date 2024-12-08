import random
import numpy as np
from collections import deque


class ReplayBuffer(deque):
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer with a fixed capacity.

        :param capacity: Maximum number of transitions to store.
        """
        super().__init__(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay buffer.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state after taking the action.
        :param done: Whether the episode is done.
        """
        self.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions.

        :param batch_size: Number of transitions to sample.
        :return: Batch as separate arrays for states, actions, rewards, next_states, and dones.
        """
        batch = random.sample(self, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )
