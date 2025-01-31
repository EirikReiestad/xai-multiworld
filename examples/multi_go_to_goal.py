from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.mdqn import MDQN

agents = 5

env = GoToGoalEnv(
    width=7,
    height=7,
    max_steps=200,
    agents=agents,
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
        eps_decay=100000,
        target_update=1000,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
)

mconfig = (
    AlgorithmConfig("DQN")
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="multi-go-to-goal", log_interval=10 * agents)
)

dqn = MDQN(agents, mconfig, config)

while True:
    dqn.learn()
