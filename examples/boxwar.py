from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.boxwar import BoxWarEnv

env = BoxWarEnv(
    width=7,
    height=7,
    max_steps=400,
    boxes=7,
    agents=6,
    team_reward=True,
    success_termination_mode="any",
)

config = (
    DQNConfig(
        batch_size=32,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=1e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000000,
        target_update=1000,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="multigrid-boxwar")
)

dqn = DQN(config)

while True:
    dqn.learn()
