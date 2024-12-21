import os
from typing import Any, Mapping

from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from utils.core.model_loader import ModelLoader
from multigrid.envs.go_to_goal import GoToGoalEnv
from xailib.core.linear_probing.linear_probe import LinearProbe

path = os.path.join("artifacts")

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

models = ModelLoader.load_models_from_path(path, dqn.model)

linear_probe = LinearProbe(dqn.model, models)
