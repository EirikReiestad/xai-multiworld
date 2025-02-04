from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.wrappers import MultiGridConceptObsWrapper
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig

env = GoToGoalEnv(
    width=7,
    height=7,
    max_steps=100,
    agents=1,
    agent_view_size=None,
    success_termination_mode="all",
    render_mode="human",
)

# env = ObservationCollectorWrapper(env, observations=10)
# concepts = None
# env = MultiGridConceptObsWrapper(
#    env, observations=200, method="random", concepts=concepts
# )

config = (
    DQNConfig(
        batch_size=64,
        replay_buffer_size=1000,
        gamma=0.99,
        learning_rate=3e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000,
        target_update=200,
    )
    .environment(env=env)
    .training()
    # .training("model_1500:v0")
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project="go-to-goal-full", log_interval=10)
)

dqn = DQN(config)

while True:
    dqn.learn()
