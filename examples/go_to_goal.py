from multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig

env = GoToGoalEnv(
    width=10,
    height=10,
    max_steps=200,
    agents=20,
    agent_view_size=9,
    success_termination_mode="all",
    render_mode="human",
)

config = (
    DQNConfig(
        batch_size=64,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=1e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=2000000,
        target_update=5000,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project="mg-go-to-goal")
)

dqn = DQN(config)

while True:
    dqn.learn()
