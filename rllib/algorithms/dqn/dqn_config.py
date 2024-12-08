from rllib.algorithms.algorithm_config import AlgorithmConfig


class DQNConfig(AlgorithmConfig):
    def __init__(self, replay_buffer_size: int = 1000, batch_size: int = 32):
        super().__init__("DQN")
        self._replay_buffer_size = replay_buffer_size
        self._batch_size = batch_size
