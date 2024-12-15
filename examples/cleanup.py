from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.cleanup import CleanUpEnv

env = CleanUpEnv(
    width=11,
    height=11,
    max_steps=1000,
    boxes=10,
    agents=10,
    success_termination_mode="any",
)

config = (
    DQNConfig(batch_size=16)
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project="multigrid-go-to-goal")
)

dqn = DQN(config)

while True:
    dqn.learn()
