from multiworld.swarm.envs.flock import FlockEnv
from rllib.algorithms.algorithm_config import AlgorithmConfig
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.mdqn import MDQN

agents = 1

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
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=3e-4,
        eps_start=0.0,
        eps_end=0.00,
        eps_decay=10000,
        target_update=1000,
    )
    .network(conv_layers=())
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
)

mconfig = (
    AlgorithmConfig("DQN")
    .model("model_800:v0")
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project="birds", log_interval=100)
)

dqn = MDQN(agents, mconfig, config)

while True:
    dqn.learn()
