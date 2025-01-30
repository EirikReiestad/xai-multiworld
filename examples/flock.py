from rllib.algorithms.ppo.ppo import PPO
from rllib.algorithms.ppo.ppo_config import PPOConfig
from swarm.envs.flock import FlockEnv

env = FlockEnv(
    width=500,
    height=500,
    max_steps=1000,
    agents=100,
    observations=10,
    predators=5,
    predator_steps=100,
    object_size=8,
    agent_view_size=65,
    success_termination_mode="all",
    render_mode="human",
    continuous_action_space=True,
)

config = (
    PPOConfig(
        batch_size=2048,
        mini_batch_size=10,
        epochs=5,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        learning_rate=3e-4,
        value_weight=0.5,
        entropy_weight=0.01,
    )
    .network(conv_layers=())
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project="mw-flockppo")
)
ppo = PPO(config)

while True:
    ppo.learn()
