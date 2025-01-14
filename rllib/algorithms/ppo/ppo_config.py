from rllib.algorithms.algorithm_config import AlgorithmConfig


class PPOConfig(AlgorithmConfig):
    def __init__(
        self,
        batch_size: int = 32,
        mini_batch_size: int = 10,
        epochs: int = 10,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        epsilon: float = 0.2,
        learning_rate: float = 1e-4,
    ):
        super().__init__("PPO")
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.learning_rate = learning_rate
