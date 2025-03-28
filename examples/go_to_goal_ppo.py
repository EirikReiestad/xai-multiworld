from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.algorithms.ppo.ppo import PPO
from rllib.algorithms.ppo.ppo_config import PPOConfig
from rllib.core.network.network import NetworkType

agents = 1
size = 7
env = GoToGoalEnv(
    width=size,
    height=size,
    static=True,
    max_steps=100,
    agents=agents,
    agent_view_size=None,
    success_termination_mode="all",
    render_mode="rgb_array",
)
config = (
    PPOConfig(
        batch_size=64,
        mini_batch_size=64,
        epochs=10,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        learning_rate=3e-4,
        value_weight=0.2,
        entropy_weight=0.2,
        continuous=False,
    )
    .network(network_type=NetworkType.MULTI_INPUT)
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    .wandb(project=f"go-to-goal-ppo-{agents}-{size}", log_interval=50)
)

ppo = PPO(config)

while True:
    ppo.learn()
