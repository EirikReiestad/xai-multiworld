from rllib.algorithms.ppo.ppo import PPO
from rllib.algorithms.ppo.ppo_config import PPOConfig
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from multigrid.envs.go_to_goal import GoToGoalEnv

env = GoToGoalEnv(
    width=4,
    height=4,
    max_steps=300,
    agents=3,
    success_termination_mode="all",
    render_mode="human",
)

config = PPOConfig().environment(env)
ppo = PPO(config)

config = DQNConfig().environment(env)
dqn = DQN(config)

while True:
    dqn.learn()
