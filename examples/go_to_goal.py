from rllib.algorithms.ppo.ppo import PPO
from rllib.algorithms.ppo.ppo_config import PPOConfig
from multigrid.envs.go_to_goal import GoToGoalEnv

env = GoToGoalEnv(
    width=10,
    height=10,
    max_steps=300,
    agents=10,
    success_termination_mode="all",
    render_mode="human",
)

config = PPOConfig().environment(env)
ppo = PPO(config)

while True:
    ppo.learn()
