# from multiworld.swarm.utils.wrappers import SwarmConceptObsWrapper
from multiworld.swarm.envs.flock import FlockEnv
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig

env = FlockEnv(
    width=200,
    height=200,
    max_steps=400,
    agents=2,
    observations=10,
    predators=1,
    predator_steps=100,
    object_size=8,
    agent_view_size=65,
    success_termination_mode="all",
    render_mode="rgb_array",
    max_predator_angle_change=45,
)

# env = ObservationCollectorWrapper(env, observations=10)
# env = SwarmConceptObsWrapper(env, observations=10, method="random")

config = (
    DQNConfig(
        batch_size=128,
        replay_buffer_size=100000,
        gamma=0.99,
        learning_rate=3e-4,
        eps_start=0.2,
        eps_end=0.05,
        eps_decay=100000,
        target_update=1000,
    )
    .network(conv_layers=())
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="birds-1v1", log_interval=100)
)

dqn = DQN(config)

while True:
    dqn.learn()
