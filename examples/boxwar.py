from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.boxwar import BoxWarEnv

env = BoxWarEnv(
    width=15,
    height=15,
    max_steps=200,
    boxes=10,
    agents=10,
    team_reward=True,
    success_termination_mode="any",
)

config = (
    DQNConfig(batch_size=16)
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="multigrid-cleanup")
)

dqn = DQN(config)

while True:
    dqn.learn()
