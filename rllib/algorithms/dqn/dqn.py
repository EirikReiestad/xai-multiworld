from rllib.algorithms.algorithm import Algorithm
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.qnetwork import QNetwork


class DQN(Algorithm):
    policy_net: QNetwork
    target_net: QNetwork

    def __init__(self, config: DQNConfig):
        super().__init__(config)
