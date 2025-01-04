import random
from collections import deque
from collections import namedtuple

Trajectory = namedtuple("Trajectory", ("state", "action", "action_prob", "reward"))


class TrajectoryBuffer(deque):
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer with a fixed capacity.

        :param capacity: Maximum number of transitions to store.
        """
        super().__init__(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        """
        Add a transition to the replay buffer.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state after taking the action.
        :param done: Whether the episode is done.
        """
        self.append(Trajectory(state, action, next_state, reward))

    def sample(self, batch_size: int) -> Trajectory:
        """
        Sample a batch of transitions.

        :param batch_size: Number of transitions to sample.
        :return: Batch as separate arrays for states, actions, rewards, next_states, and dones.
        """
        return random.sample(self, batch_size)

    def reset(self):
        self.clear()
