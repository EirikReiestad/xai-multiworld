from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.cleanup import CleanUpEnv

env = CleanUpEnv(
    width=7,
    height=7,
    max_steps=200,
    boxes=10,
    agents=6,
    success_termination_mode="any",
)

config = (
    DQNConfig(batch_size=16)
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project="multigrid-cleanup")
)

dqn = DQN(config)

while True:
    dqn.learn()
