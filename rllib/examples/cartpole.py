import gymnasium as gym
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.wrappers.single_algorithm import SingleAlgorithm

cartpole = gym.make("CartPole-v1", render_mode="human")

dqn_config = DQNConfig().environment(env=cartpole)
dqn = DQN(dqn_config)
dqn = SingleAlgorithm(dqn)

dqn.learn()
