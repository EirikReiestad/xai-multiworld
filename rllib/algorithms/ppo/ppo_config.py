from rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib.core.network.network import NetworkType


class PPOConfig(AlgorithmConfig):
    def __init__(
        self,
        batch_size: int = 32,
        mini_batch_size: int = 10,  # Depricated
        epochs: int = 10,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        epsilon: float = 0.2,
        learning_rate: float = 1e-4,
        value_weight: float = 0.5,
        entropy_weight: float = 0.01,
        continuous: bool = True,
    ):
        super().__init__("PPO")
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.network(network_type=NetworkType.FEED_FORWARD)
        self.continuous = continuous
