from rllib.algorithms.ppo.ppo import PPO
from rllib.algorithms.ppo.ppo_config import PPOConfig
from multigrid.envs.go_to_goal import GoToGoalEnv

env = GoToGoalEnv(
    width=7,
    height=7,
    max_steps=4,
    agents=3,
    success_termination_mode="all",
    render_mode="human",
)

config = (
    PPOConfig(
        batch_size=32,
        mini_batch_size=16,
        epochs=3,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        learning_rate=1e-3,
    )
    .environment(env)
    .training()
    .debugging(log_level="INFO")
    .rendering()
    #     .wandb(project="test", log_interval=1)
)
ppo = PPO(config)

while True:
    ppo.learn()
