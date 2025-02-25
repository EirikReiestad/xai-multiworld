import logging
from typing import Dict

import numpy as np
from scipy.spatial.distance import pdist
from torch._dynamo.utils import tabulate

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


def run(concept: str):
    ignore = ["_fc0"]
    layer_idx = 4

    model = ModelLoader.load_latest_model_from_path("artifacts", dqn.model)
    models = {"latest": model}
    positive_observation, test_observation = load_and_split_observation(concept, 1.0)
    negative_observation, _ = load_and_split_observation("negative_" + concept, 1.0)

    probes, positive_activations, negative_activations = get_probes(
        models, positive_observation, negative_observation, ignore
    )

    layer_activations = list(positive_activations["latest"].values())[layer_idx][
        "output"
    ]
    probe = list(probes["latest"].values())[layer_idx]

    stats = calculate_statistics(layer_activations.detach().numpy())

    return stats


def calculate_statistics(points: np.ndarray):
    n_samples = points.shape[0]
    points = points.reshape(n_samples, -1)

    mean = np.mean(points)
    variance = np.var(points)
    std_dev = np.std(points)
    median = np.median(points)
    range_val = np.ptp(points)

    q75, q25 = np.percentile(points, [75, 25])
    iq_range = q75 - q25

    """
    scaler = StandardScaler()
    points = scaler.fit_transform(points)
    kde = KernelDensity(kernel="gaussian")
    kde.fit(points)
    log_density = kde.score_samples(points)
    density = np.exp(log_density)
    """

    pairwise_distances = pdist(points, "euclidean")
    mean_distance = np.mean(pairwise_distances)
    density = 1 / mean_distance if mean_distance != 0 else float("inf")

    return {
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "median": median,
        "range": range_val,
        "iq_range": iq_range,
        "density": density,
    }


def log_stats(stats: Dict[str, Dict[str, float]]):
    metrics = list(next(iter(stats.values())).keys())
    concepts = list(stats.keys())

    table_data = []
    for metric in metrics:
        row = [metric] + [stats[concept][metric] for concept in concepts]
        table_data.append(row)

    table = tabulate(
        table_data,
        headers=["Metric"] + concepts,
    )
    logging.info(f"\n{table}")


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

    stats = {}

    for concept in concepts:
        stat = run(concept)
        stats[concept] = stat

    log_stats(stats)
