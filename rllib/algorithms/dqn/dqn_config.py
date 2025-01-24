from rllib.algorithms.algorithm_config import AlgorithmConfig
from typing import Tuple


class DQNConfig(AlgorithmConfig):
    def __init__(
        self,
        replay_buffer_size: int = 10000,
        batch_size: int = 16,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: int = 1000,
        target_update: int = 1000,
        conv_layers: Tuple[int, ...] = (32, 64, 64),
        hidden_units: Tuple[int, ...] = (128, 128),
    ):
        super().__init__("DQN")
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
