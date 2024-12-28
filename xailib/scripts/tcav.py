from multigrid.envs.go_to_goal import GoToGoalEnv
from xailib.core.plotting.plot3d import plot_3d
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from utils.common.observation import (
    zip_observation_data,
    load_and_split_observation,
)
from utils.core.model_loader import ModelLoader
from xailib.common.activations import (
    compute_activations_from_artifacts,
)
from xailib.common.tcav_score import tcav_scores
from xailib.common.probes import get_probes

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

concept = "random"

model_artifacts = ModelLoader.load_models_from_path("artifacts", dqn.model)
positive_observation, test_observation = load_and_split_observation(concept, 0.8)
negative_observation, _ = load_and_split_observation("random_negative", 0.8)

probes = get_probes(model_artifacts, positive_observation, negative_observation)

test_observation_zipped = zip_observation_data(test_observation)

test_activations, test_input, test_output = compute_activations_from_artifacts(
    model_artifacts, test_observation_zipped
)

tcav_scores = tcav_scores(test_activations, test_output, probes)

plot_3d(
    tcav_scores,
    label=concept,
    filename="concept_score",
    min=0,
    max=1,
)
