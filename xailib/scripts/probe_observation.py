import os

import torch

from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.observation import pertubate_observation
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from utils.common.observation import (
    Observation,
    load_and_split_observation,
    observation_from_dict,
    observation_from_file,
    observation_to_file,
    zip_observation_data,
    zipped_torch_observation_data,
)
from utils.core.model_loader import ModelLoader
from xailib.common.activations import (
    compute_activations_from_models,
)
from xailib.common.probes import get_probes
from xailib.core.linear_probing.probe_to_observation import (
    maximize_activation,
)


def run(concept: str):
    ignore = []

    model = ModelLoader.load_latest_model_from_path("artifacts", dqn.model)
    models = {"latest": model}
    positive_observation, test_observation = load_and_split_observation(concept, 0.8)
    negative_observation, _ = load_and_split_observation("random_negative", 0.8)

    test_observation_zipped = zip_observation_data(test_observation)

    test_activations, test_input, test_output = compute_activations_from_models(
        models, test_observation_zipped, ignore
    )

    probes = get_probes(models, positive_observation, negative_observation, ignore)

    layer_idx = 4
    probe = list(probes["latest"].values())[layer_idx]
    activation = list(list(test_activations.values())[0].values())[layer_idx]["output"][
        0
    ]

    observations = observation_from_file(
        os.path.join("assets", "concepts", "random.json")
    )
    observations_zipped = zip_observation_data(observations)
    observation_activations, obs_input, obs_output = compute_activations_from_models(
        models, observations_zipped, ignore
    )

    activations = list(observation_activations["latest"].values())[layer_idx]["output"]

    similarities = []
    for activation in activations:
        cos = torch.nn.CosineSimilarity()
        coef = torch.Tensor(probe.coef_)
        similarity = cos(activation.flatten(), coef)
        similarities.append(abs(similarity))

    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i])
    sorted_observations = observations[sorted_indices][..., Observation.DATA]
    observation_to_file(
        sorted_observations,
        os.path.join("assets", "concepts", "sorted_" + concept + ".json"),
    )


if __name__ == "__main__":
    import os

    # os.chdir("../../")
    artifact = ModelLoader.load_latest_model_artifacts_from_path("artifacts")

    width = artifact.metadata.get("width")
    height = artifact.metadata.get("height")
    agents = artifact.metadata.get("agents")
    conv_layers = artifact.metadata.get("conv_layers")
    hidden_units = artifact.metadata.get("hidden_units")
    eps_threshold = artifact.metadata.get("eps_threshold")
    learning_rate = artifact.metadata.get("learning_rate")

    env = GoToGoalEnv(
        width=width, height=height, agents=agents, render_mode="rgb_array"
    )

    config = (
        DQNConfig(
            batch_size=128,
            replay_buffer_size=10000,
            gamma=0.99,
            learning_rate=3e-4,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=50000,
            target_update=1000,
        )
        .network(
            network_type=NetworkType.MULTI_INPUT,
            conv_layers=conv_layers,
            hidden_units=hidden_units,
        )
        .debugging(log_level="INFO")
        .environment(env=env)
    )

    dqn = DQN(config)

    # concepts = concept_checks.keys()
    concepts = ["random"]
    concepts = [
        "random",
        "agent_in_front",
        "agent_in_view",
        "agent_to_left",
        "agent_to_right",
        "goal_in_front",
        "goal_in_view",
        "goal_to_left",
        "goal_to_right",
    ]

    for concept in concepts:
        run(concept)
