from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.cleanup import CleanUpEnv

env = CleanUpEnv(
    width=7,
    height=7,
    max_steps=300,
    boxes=2,
    agents=10,
    success_termination_mode="all",
)

config = (
    DQNConfig(batch_size=8)
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project="multigrid-go-to-goal")
)

dqn = DQN(config)

while True:
    dqn.learn()
