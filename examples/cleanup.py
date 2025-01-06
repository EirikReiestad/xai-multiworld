from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.cleanup import CleanUpEnv

env = CleanUpEnv(
    width=8,
    height=8,
    max_steps=250,
    boxes=4,
    agents=3,
    success_termination_mode="any",
)

config = (
    DQNConfig(
        batch_size=32,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=1e-4,
        eps_start=0.2,
        eps_end=0.05,
        eps_decay=500000,
        target_update=1000,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="multigrid-cleanupv0")
)

dqn = DQN(config)

while True:
    dqn.learn()
