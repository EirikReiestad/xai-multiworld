from matplotlib.pyplot import step
from torch.optim import lr_scheduler
from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType

env = GoToGoalEnv(
    goals=1,
    width=10,
    height=10,
    max_steps=100,
    agents=1,
    agent_view_size=7,
    success_termination_mode="all",
    render_mode="human",
)

# env = ObservationCollectorWrapper(env, observations=10)
# concepts = None
#    env, observations=200, method="random", concepts=concepts
# env = MultiGridConceptObsWrapper(
# )

config = (
    DQNConfig(
        batch_size=128,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=3e-4,
        eps_start=0.0,
        eps_end=0.00,
        eps_decay=1000,
        update_method="soft",
        target_update=100,
    )
    .network(network_type=NetworkType.MULTI_INPUT)
    .environment(env=env)
    .model(model="model_5000:v0")
    # .training()
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project="test", log_interval=20)
)

dqn = DQN(config)

while True:
    dqn.learn()
