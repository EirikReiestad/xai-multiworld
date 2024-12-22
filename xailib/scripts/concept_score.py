import os

from multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from utils.core.model_loader import ModelLoader
from xailib.common.probes import get_probes
from utils.common.observation import observation_from_file
from xailib.common.binary_concept_score import binary_concept_score

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

positive_concept = "goal"
negative_concept = "random"

model_path = os.path.join("artifacts")
model_artifacts = ModelLoader.load_models_from_path(model_path, dqn.model)

concept_path = os.path.join("assets", "concepts")

positive_observation = observation_from_file(
    os.path.join(concept_path, positive_concept + ".json")
)
negative_observation = observation_from_file(
    os.path.join(concept_path, negative_concept + ".json")
)
probes = get_probes(model_artifacts, positive_observation, negative_observation)
print(probes)
