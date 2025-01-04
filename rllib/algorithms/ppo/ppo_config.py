from rllib.algorithms.algorithm_config import AlgorithmConfig


class PPOConfig(AlgorithmConfig):
    def __init__(
        self,
        batch_size: int = 64,
        max_grad_norm: float = 1.0,
        epochs: int = 10,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        entropy_eps: float = 1e-4,
        learning_rate: float = 1e-4,
        target_update: int = 10,
    ):
        super().__init__("PPO")
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.learning_rate = learning_rate
        self.target_update = target_update
