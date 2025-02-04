from multiworld.swarm.envs.flock import FlockEnv
from rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.mdqn import MDQN

agents = 10

env = FlockEnv(
    width=200,
    height=200,
    max_steps=400,
    agents=agents,
    observations=10,
    predators=1,
    predator_steps=100,
    object_size=8,
    agent_view_size=65,
    success_termination_mode="all",
    render_mode="human",
)

config = (
    DQNConfig(
        batch_size=64,
        replay_buffer_size=100000,
        gamma=0.99,
        learning_rate=3e-4,
        eps_start=0.2,
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
    .wandb(project="birds", log_interval=10 * agents)
)

dqn = MDQN(agents, mconfig, config)

while True:
    dqn.learn()
