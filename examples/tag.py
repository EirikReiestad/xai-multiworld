from multiworld.multigrid.envs.tag import TagEnv
from rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.mdqn import MDQN
from rllib.core.network.network import NetworkType

agents = 5

env = TagEnv(
    width=7,
    height=7,
    max_steps=200,
    agents=agents,
    success_termination_mode="all",
    render_mode="rgb_array",
)

config = (
    DQNConfig(
        batch_size=128,
        replay_buffer_size=100000,
        gamma=0.99,
        learning_rate=3e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=50000,
        target_update=200,
    )
    .network(network_type=NetworkType.MULTI_INPUT)
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
)

mconfig = (
    AlgorithmConfig("DQN")
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="multi-tag-1", log_interval=100)
)

dqn = MDQN(agents, mconfig, config)

while True:
    dqn.learn()
