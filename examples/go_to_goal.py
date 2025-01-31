from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig

env = GoToGoalEnv(
    width=15,
    height=15,
    max_steps=200,
    agents=20,
    agent_view_size=5,
    success_termination_mode="all",
    render_mode="rgb_array",
)


config = (
    DQNConfig(
        batch_size=64,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=3e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=10000,
        target_update=500,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="go-to-goal")
)

dqn = DQN(config)

while True:
    dqn.learn()
