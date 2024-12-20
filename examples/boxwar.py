from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.boxwar import BoxWar

env = BoxWar(
    width=5,
    height=5,
    max_steps=201,
    boxes=6,
    agents=2,
    team_reward=True,
    success_termination_mode="all",
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
