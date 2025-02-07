import gymnasium as gym
from torch.optim import lr_scheduler

from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from rllib.utils.wrappers import GymnasiumWrapper

env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()

env = GymnasiumWrapper(env)

config = (
    DQNConfig(
        batch_size=128,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=3e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000,
        update_method="hard",
        target_update=200,
    )
    .network(
        conv_layers=(), hidden_units=(128, 128), network_type=NetworkType.FEED_FORWARD
    )
    .environment(env=env)
    .training(lr_scheduler="cyclic")
    # .training("model_1500:v0")
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="CartPole-v1", log_interval=10)
)

dqn = DQN(config)

while True:
    dqn.learn()
