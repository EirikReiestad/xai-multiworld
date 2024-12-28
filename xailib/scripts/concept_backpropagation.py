import os

from multigrid.envs.go_to_goal import GoToGoalEnv
from xailib.core.plotting.plot3d import plot_3d
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from utils.common.observation import (
    observation_from_file,
    split_observation,
    zip_observation_data,
    set_require_grad,
)
from utils.core.model_loader import ModelLoader
from xailib.common.activations import ActivationTracker
from xailib.common.tcav_score import tcav_scores
from xailib.common.probes import get_probes
from xailib.common.concept_backpropagation import feature_concept_importance

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
negative_concept = "random_negative"

model_path = os.path.join("artifacts")
model_artifacts = ModelLoader.load_models_from_path(model_path, dqn.model)

concept_path = os.path.join("assets", "concepts")

positive_observation = observation_from_file(
    os.path.join(concept_path, positive_concept + ".json")
)
positive_observation, test_observation = split_observation(positive_observation, 0.8)

negative_observation = observation_from_file(
    os.path.join(concept_path, negative_concept + ".json")
)
negative_observation, _ = split_observation(negative_observation, 0.8)

probes = get_probes(model_artifacts, positive_observation, negative_observation)

test_observation_zipped = zip_observation_data(test_observation)
set_require_grad(test_observation_zipped)

test_activations = {}
test_input = {}
test_output = {}
for key, value in model_artifacts.items():
    activations, input, output = ActivationTracker(value.model).compute_activations(
        test_observation_zipped
    )
    test_activations[key] = activations
    test_input[key] = input
    test_output[key] = output

feature_concept_importance = feature_concept_importance(
    test_activations, test_input, probes
)
tcav_scores = tcav_scores(test_activations, test_output, probes)

plot_3d(
    tcav_scores,
    label=positive_concept,
    filename="concept_score",
    min=0,
    max=1,
)
