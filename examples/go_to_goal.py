from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn import DQNConfig
from multigrid.envs.go_to_goal import GoToGoalEnv

env = GoToGoalEnv(
    width=20,
    height=20,
    max_steps=300,
    agents=20,
    success_termination_mode="all",
    render_mode="rgb_array",
)

config = (
    DQNConfig(
        batch_size=64,
        replay_buffer_size=100000,
        gamma=0.999,
        learning_rate=1e-4,
        eps_start=0.2,
        eps_end=0.05,
        eps_decay=500000,
        target_update=2000,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="multigrid-cleanup")
)

dqn = DQN(config)

while True:
    dqn.learn()
