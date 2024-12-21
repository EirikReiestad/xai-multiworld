import os

from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from utils.core.model_loader import ModelLoader
from multigrid.envs.go_to_goal import GoToGoalEnv
from utils.common.observation import Observation, observation_from_file

env = GoToGoalEnv(render_mode="rgb_array")

config = (
    DQNConfig(
        batch_size=16,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=1e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=50000,
        target_update=1000,
    )
    .debugging(log_level="INFO")
    .environment(env=env)
)

dqn = DQN(config)

path = os.path.join("artifacts", "models")
models = ModelLoader.load_models_from_path(path, dqn.model)

concept = "goal"
path = os.path.join("artifacts", "concepts", concept + ".json")
goal_observation = observation_from_file(path)

print(goal_observation)

"""
linear_probe = LinearProbe(
    dqn.model,
    models,
    observations.positive_observation,
    observations.negative_observation,
)
"""
