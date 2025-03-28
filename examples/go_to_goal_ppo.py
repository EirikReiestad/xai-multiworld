from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.algorithms.ppo.ppo import PPO
from rllib.algorithms.ppo.ppo_config import PPOConfig
from rllib.core.network.network import NetworkType

agents = 1
env = GoToGoalEnv(
    width=7,
    height=7,
    static=True,
    max_steps=100,
    agents=agents,
    agent_view_size=7,
    success_termination_mode="all",
    render_mode="human",
)
config = (
    PPOConfig(
        batch_size=100,
        mini_batch_size=100,
        epochs=5,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        learning_rate=3e-4,
        value_weight=0.5,
        entropy_weight=0.1,
        continuous=False,
    )
    .network(network_type=NetworkType.MULTI_INPUT)
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    # .wandb(project=f"go-to-goal-ppo-{agents}", log_interval=20)
)

ppo = PPO(config)

while True:
    ppo.learn()
