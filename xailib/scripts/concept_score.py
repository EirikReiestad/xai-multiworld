from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from utils.common.observation import (
    load_and_split_observation,
    zip_observation_data,
)
from utils.core.model_loader import ModelLoader
from utils.core.plotting import plot_3d
from xailib.common.activations import (
    compute_activations_from_models,
)
from xailib.common.concept_score import binary_concept_scores
from xailib.common.probes import get_probes


def run(concept: str):
    ignore = ["_fc0"]

    model_artifacts = ModelLoader.load_models_from_path("artifacts", dqn.model)
    positive_observation, test_observation = load_and_split_observation(concept, 0.8)
    negative_observation, _ = load_and_split_observation("negative_" + concept, 0.8)

    test_observation_zipped = zip_observation_data(test_observation)

    test_activations, test_input, test_output = compute_activations_from_models(
        model_artifacts, test_observation_zipped, ignore
    )

    probes, positive_activations, negative_activations = get_probes(
        model_artifacts, positive_observation, negative_observation, ignore
    )

    concept_scores = binary_concept_scores(test_activations, probes)

    plot_3d(
        concept_scores,
        label=concept,
        filename=concept,
        min=0,
        max=1,
        show=True,
    )


if __name__ == "__main__":
    artifact = ModelLoader.load_latest_model_artifacts_from_path("artifacts")

    width = artifact.metadata.get("width")
    height = artifact.metadata.get("height")
    agents = artifact.metadata.get("agents")
    conv_layers = artifact.metadata.get("conv_layers")
    hidden_units = artifact.metadata.get("hidden_units")
    eps_threshold = artifact.metadata.get("eps_threshold")
    learning_rate = artifact.metadata.get("learning_rate")

    env = GoToGoalEnv(
        width=width,
        height=height,
        agents=agents,
        preprocessing=PreprocessingEnum.ohe_minimal,
        render_mode="rgb_array",
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
        "goal_in_front",
        "goal_in_view",
        "goal_to_left",
        "goal_to_right",
        "wall_in_view",
        # "agent_in_front",
        # "agent_in_view",
        # "agent_to_left",
        # "agent_to_right",
    ]

    for concept in concepts:
        run(concept)
