from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig

env = GoToGoalEnv(
    width=20,
    height=20,
    max_steps=200,
    agents=20,
    agent_view_size=7,
    success_termination_mode="all",
    render_mode="rgb_array",
)

# env = ObservationCollectorWrapper(env, observations=10)
# env = MultiGridConceptObsWrapper(
#    env, observations=10, method="random", concepts=["random"]
# )

config = (
    DQNConfig(
        batch_size=64,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=3e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=100000,
        target_update=400,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="go-to-goalv4", log_interval=100)
)

dqn = DQN(config)

while True:
    dqn.learn()
