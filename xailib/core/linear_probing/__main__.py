import os

from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from utils.core.model_loader import ModelLoader
from multigrid.envs.go_to_goal import GoToGoalEnv
from utils.common.observation import Observation, observation_from_file
from xailib.core.linear_probing.linear_probe import LinearProbe

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

path = os.path.join("artifacts")
model_artifacts = ModelLoader.load_models_from_path(path, dqn.model)
model_artifact = next(iter(model_artifacts.values()))

concepts = ["goal", "random"]
observations = {
    concept: observation_from_file(
        os.path.join("assets", "concepts", concept + ".json")
    )
    for concept in concepts
}

observations["goal"][..., Observation.LABEL] = 1
observations["random"][..., Observation.LABEL] = 0

linear_probe = LinearProbe(
    dqn.model,
    observations["goal"],
    observations["random"],
)
linear_probe.train()
