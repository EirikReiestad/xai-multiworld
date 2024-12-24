from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.wrappers.dqn_concept_bottleneck_wrapper import DQNConceptBottleneckWrapper

env = GoToGoalEnv(
    width=10,
    height=10,
    max_steps=100,
    agents=10,
    success_termination_mode="all",
    render_mode="rgb_array",
)

config = (
    DQNConfig(
        batch_size=16,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=1e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=50000,
        target_update=1000,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project="multigrid-go-to-goal-cbm")
)

dqn = DQN(config)
dqn = DQNConceptBottleneckWrapper(dqn)

while True:
    dqn.learn()
