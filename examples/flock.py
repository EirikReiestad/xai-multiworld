from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from multiworld.envs.flock import FlockEnv

env = FlockEnv(
    width=1000,
    height=1000,
    max_steps=1000,
    agents=100,
    success_termination_mode="all",
    render_mode="human",
    object_size=8,
    agent_view_size=101,
)

config = (
    DQNConfig(
        batch_size=32,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=1e-4,
        eps_start=0.2,
        eps_end=0.05,
        eps_decay=100000,
        target_update=1000,
    )
    .network(conv_layers=())
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project="multigrid-cleanupv0")
)

dqn = DQN(config)

while True:
    dqn.learn()
