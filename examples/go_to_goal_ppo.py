from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.algorithms.ppo.ppo import PPO
from rllib.algorithms.ppo.ppo_config import PPOConfig
from rllib.core.network.network import NetworkType

env = GoToGoalEnv(
    width=5,
    height=5,
    max_steps=100,
    agents=1,
    agent_view_size=None,
    success_termination_mode="all",
    render_mode="rgb_array",
)
config = (
    PPOConfig(
        batch_size=200,
        mini_batch_size=10,
        epochs=5,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        learning_rate=3e-4,
        value_weight=0.5,
        entropy_weight=0.01,
        continuous=False,
    )
    .network(network_type=NetworkType.MULTI_INPUT)
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project="go-to-goal-ppo", log_interval=20)
)

ppo = PPO(config)

while True:
    ppo.learn()
