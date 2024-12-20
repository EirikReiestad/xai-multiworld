from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.boxwar import BoxWarEnv

env = BoxWarEnv(
    width=10,
    height=10,
    max_steps=200,
    boxes=6,
    agents=2,
    team_reward=True,
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
