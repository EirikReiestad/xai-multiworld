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
from xailib.common.probes import get_probes
from xailib.common.tcav_score import tcav_scores


def run(concept: str):
    ignore = ["_fc0"]

    models = ModelLoader.load_models_from_path("artifacts", dqn.model)
    positive_observation, test_observation = load_and_split_observation(concept, 0.8)
    negative_observation, _ = load_and_split_observation("negative_" + concept, 0.8)

    probes, positive_activations, negative_activations = get_probes(
        models, positive_observation, negative_observation, ignore
    )

    test_observation_zipped = zip_observation_data(test_observation)

    test_activations, test_input, test_output = compute_activations_from_models(
        models, test_observation_zipped, ignore
    )

    scores = tcav_scores(test_activations, test_output, probes)

    plot_3d(
        scores,
        label=concept,
        filename="tcav_" + concept,
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
        width=10,
        height=10,
        agents=1,
        preprocessing=PreprocessingEnum.ohe_minimal,
    )

    config = (
        DQNConfig()
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
        # "agent_in_front",
        # "agent_in_view",
        # "agent_to_left",
        # "agent_to_right",
        "goal_in_front",
        "goal_in_view",
        "goal_to_left",
        "goal_to_right",
        "wall_in_view",
    ]

    for concept in concepts:
        run(concept)
