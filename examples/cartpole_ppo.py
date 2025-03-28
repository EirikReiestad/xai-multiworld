import gymnasium as gym

from rllib.algorithms.ppo.ppo import PPO
from rllib.algorithms.ppo.ppo_config import PPOConfig
from rllib.core.network.network import NetworkType
from rllib.utils.wrappers import GymnasiumWrapper

env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()

env = GymnasiumWrapper(env)

config = (
    PPOConfig(
        batch_size=64,
        epochs=10,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        learning_rate=3e-4,
        value_weight=0.5,
        entropy_weight=0.01,
        continuous=False,
    )
    .network(
        conv_layers=(), hidden_units=(128, 128), network_type=NetworkType.FEED_FORWARD
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="CartPole-v1", log_interval=10)
)
ppo = PPO(config)

while True:
    ppo.learn()
