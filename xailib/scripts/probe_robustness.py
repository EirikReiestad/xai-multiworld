import logging
from typing import Dict, List

from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from utils.common.observation import (
    load_and_split_observation,
)
from utils.core.model_loader import ModelLoader
from xailib.common.probes import get_probes


def run(concepts: List[str], layer_idx: int = 2):
    splits = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]

    concept_similarities = {}
    for concept in concepts:
        similarities = get_probe_similarity(concept, layer_idx, splits)
        concept_similarities[concept] = similarities

    log_similarity(concept_similarities)

    return concept_similarities


def log_similarity(similarities: Dict):
    y_axis_values = sorted(next(iter(similarities.values())).keys(), reverse=True)

    table_data = []
    for y in y_axis_values:
        row = [y] + [similarities[key][y][0][0] for key in similarities.keys()]
        table_data.append(row)

    headers = ["dataset size"] + concepts
    logging.info("\n" + tabulate(table_data, headers=headers))


def get_probe_similarity(
    concept: str, layer_idx: int, splits: List[float]
) -> Dict[float, float]:
    ignore = ["_fc0"]

    base = get_probe(concept, layer_idx, 1.0)

    probe_splits = {}
    for split in splits:
        probe = get_probe(concept, layer_idx, split)
        probe_splits[split] = probe

    similarities = {
        key: cosine_similarity(base.coef_, probe.coef_)
        for key, probe in probe_splits.items()
    }
    return similarities


def get_probe(concept: str, layer_idx: int, split: float = 0.8):
    ignore = []

    model = ModelLoader.load_latest_model_from_path("artifacts", dqn.model)
    models = {"latest": model}
    positive_observation, test_observation = load_and_split_observation(concept, split)
    negative_observation, _ = load_and_split_observation("negative_" + concept, split)

    probes, positive_activations, negative_activations = get_probes(
        models, positive_observation, negative_observation, ignore
    )

    probe = list(probes["latest"].values())[layer_idx]

    return probe


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

    run(concepts)
