from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType

agents = 15
size = 20
env = GoToGoalEnv(
    goals=1,
    static=True,
    width=size,
    height=size,
    max_steps=100,
    preprocessing=PreprocessingEnum.ohe_minimal,
    highlight=True,
    agents=agents,
    agent_view_size=7,
    success_termination_mode="all",
    render_mode="rgb_array",
)

config = (
    DQNConfig(
        batch_size=128,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=3e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=5000,
        update_method="soft",
        target_update=100,
    )
    .network(network_type=NetworkType.MULTI_INPUT)
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project=f"go-to-goal-{agents}-{size}", log_interval=100)
)

dqn = DQN(config)

while True:
    dqn.learn()
