from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from multiworld.envs.flock import FlockEnv

env = FlockEnv(
    width=1000,
    height=1000,
    max_steps=1000,
    agents=100,
    object_size=8,
    agent_view_size=65,
    success_termination_mode="all",
    render_mode="rgb_array",
)

config = (
    DQNConfig(
        batch_size=128,
        replay_buffer_size=100000,
        gamma=0.99,
        learning_rate=1e-4,
        eps_start=0.2,
        eps_end=0.05,
        eps_decay=2000000,
        target_update=5000,
    )
    .network(conv_layers=())
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="mw-flock")
)

dqn = DQN(config)

while True:
    dqn.learn()
