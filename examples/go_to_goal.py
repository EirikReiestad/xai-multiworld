from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.go_to_goal import GoToGoalEnv
from multigrid.wrappers import ConceptObsWrapper

env = GoToGoalEnv(
    width=10,
    height=10,
    max_steps=300,
    agents=10,
    success_termination_mode="all",
    render_mode="rgb_array",
)
env_wrapped = ConceptObsWrapper(
    env, observations=1000, concepts=["random"], method="random"
)

config = (
    DQNConfig(
        batch_size=32,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=1e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=100000,
        target_update=1000,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="dell-multigrid-go-to-goal")
)

dqn = DQN(config)

while True:
    dqn.learn()
