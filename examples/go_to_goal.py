from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn import DQNConfig
from multigrid.envs.go_to_goal import GoToGoalEnv

env = GoToGoalEnv(
    width=30,
    height=30,
    max_steps=300,
    agents=40,
    success_termination_mode="all",
    render_mode="rgb_array",
    agent_view_size=11,
)

config = (
    DQNConfig(
        batch_size=128,
        replay_buffer_size=100000,
        gamma=0.999,
        learning_rate=1e-4,
        eps_start=0.2,
        eps_end=0.05,
        eps_decay=1000000,
        target_update=5000,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="mg-cleanup")
)

dqn = DQN(config)

while True:
    dqn.learn()
