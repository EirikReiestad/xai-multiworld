from rllib.algorithms.algorithm_config import AlgorithmConfig


class PPOConfig(AlgorithmConfig):
    def __init__(
        self,
        batch_size: int = 64,
        max_grad_norm: float = 1.0,
        num_epochs: int = 10,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        entropy_eps: float = 1e-4,
        learning_rate: float = 1e-4,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: int = 1000,
        target_update: int = 1000,
    ):
        super().__init__("PPO")
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
