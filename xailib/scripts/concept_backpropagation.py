import numpy as np

from multigrid.core.agent import Agent
from multigrid.core.constants import TILE_PIXELS
from multigrid.core.grid import Grid
from multigrid.envs.go_to_goal import GoToGoalEnv
from multigrid.utils.observation import agents_from_agent_observation
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from utils.common.observation import (
    load_and_split_observation,
    set_require_grad,
    zip_observation_data,
)
from utils.core.model_loader import ModelLoader
from xailib.common.activations import compute_activations_from_artifacts
from xailib.common.concept_backpropagation import feature_concept_importance
from xailib.common.probes import get_probes
from xailib.common.tcav_score import tcav_scores
from utils.common.image import normalize_image
from utils.common.element_matrix import (
    images_to_element_matrix,
    flatten_element_matrices,
)
from utils.common.numpy_collections import remove_nan
from utils.core.plotting import show_image, box_plot, plot_heatmap
from multigrid.core.world_object import WorldObject
from multigrid.core.constants import WorldObjectType, Color

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

concept = "goal"
layer = 0

model_artifacts = ModelLoader.load_latest_model_from_path("artifacts", dqn.model)
positive_observation, test_observation = load_and_split_observation(concept, 0.8)
negative_observation, _ = load_and_split_observation("random_negative", 0.8)

probes = get_probes(model_artifacts, positive_observation, negative_observation)

test_observation_zipped = zip_observation_data(test_observation)
set_require_grad(test_observation_zipped)

test_activations, test_input, test_output = compute_activations_from_artifacts(
    model_artifacts, test_observation_zipped
)

last_layer_activations = next(reversed(test_activations["latest"].values()))["output"]

grads_img, grads_dir = feature_concept_importance(
    last_layer_activations, test_input["latest"]
)

for grad, obs in zip(grads_img, test_observation):
    numpy_grad = grad.detach().numpy()
    grid_obs = np.array(obs.features[0]["image"])
    agents = agents_from_agent_observation(grid_obs)
    agent = Agent(0)
    agent.pos = (grid_obs.shape[1] // 2, grid_obs.shape[0] - 1)
    agent.dir = 3
    agents.append(agent)
    grid = Grid.from_numpy(grid_obs)
    img = grid.render(TILE_PIXELS, agents=agents)

    grad_sum = numpy_grad.sum(axis=2)  # sum of gradients for the observation object
    plot_heatmap(grad_sum, background=img, alpha=0.5, title=concept, show=False)

image_matrices = images_to_element_matrix(
    grads_img.detach().numpy(), test_observation, average=False, absolute=True
)
elements = flatten_element_matrices(image_matrices)

for key, value in elements.items():
    value = remove_nan(value)
    if key[WorldObject.TYPE] == WorldObjectType.unseen.to_index():
        title = "unseen"
    elif key[WorldObject.TYPE] == WorldObjectType.agent.to_index():
        title = "agent"
    else:
        title = str(WorldObject.from_array(key))
    values = [_ for _ in zip(*value)]
    box_plot(values, title, labels=["type", "color", "state"], y_label="grads")

image_matrices = images_to_element_matrix(
    grads_img.detach().numpy(), test_observation, absolute=True
)
normalized_images = {
    key: normalize_image(value) for key, value in image_matrices.items()
}

for key, value in normalized_images.items():
    if key[WorldObject.TYPE] == WorldObjectType.unseen.to_index():
        title = "unseen"
    elif key[WorldObject.TYPE] == WorldObjectType.agent.to_index():
        title = "agent"
    else:
        title = str(WorldObject.from_array(key))
    show_image(value, title, rgb=True, rgb_titles=("type", "color", "state"))

tcav_scores = tcav_scores(test_activations, test_output, probes)
