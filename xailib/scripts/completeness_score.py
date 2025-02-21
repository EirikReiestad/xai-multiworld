import ast
import json
import logging
import math
import os
from collections import defaultdict
from itertools import count
from typing import Dict, List, Literal

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate
from torch.utils.data import TensorDataset

from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from utils.common.collections import get_combinations
from utils.common.observation import (
    Observation,
    load_and_split_observation,
    observations_from_file,
    zip_observation_data,
)
from utils.core.model_loader import ModelLoader
from xailib.common.activations import (
    compute_activations_from_models,
)
from xailib.common.concept_score import (
    individual_binary_concept_score,
    individual_soft_concept_score,
)
from xailib.common.probes import get_probes
from xailib.common.train_model import train_model
from xailib.core.network.feed_forward import FeedForwardNetwork

logging.basicConfig(level=logging.INFO)


def run(concepts: List[str], dqn: DQN, layer_idx: int = 2):
    ignore = ["_fc0"]
    probes = {}
    for concept in concepts:
        probe = get_probe(concept, layer_idx)
        probes[concept] = probe

    observations = observations_from_file(
        os.path.join("assets/observations", "observations" + ".json")
    )

    model = ModelLoader.load_latest_model_from_path("artifacts", dqn.model)
    models = {"latest": model}
    action_space = int(dqn.action_space.n)

    observation_zipped = zip_observation_data(observations)
    activations, input, output = compute_activations_from_models(
        models, observation_zipped, ignore
    )

    random_probe = {"random": get_probe("random", layer_idx)}
    random_loss, random_accuracy = compute_accuracy(
        1, action_space, activations, observations, random_probe, layer_idx
    )
    logging.info(f"Random concept - loss: {random_loss} accuracy: {random_accuracy}")

    results = {tuple(["random"]): (random_loss, random_accuracy)}
    for comb in get_combinations(concepts):
        logging.info(f"\n\n===== Computing accuracy for {comb} =====")
        sub_probes = {key: value for key, value in probes.items() if key in comb}
        sub_loss, sub_accuracy = compute_accuracy(
            len(comb), action_space, activations, observations, sub_probes, layer_idx
        )
        results[tuple(sorted(comb))] = (sub_loss, sub_accuracy)

    path = os.path.join(
        "assets", "results", f"concept_combination_accuracies_{layer_idx}.json"
    )

    write_results(results, path)

    calculate_shapley_values(path)


def calculate_shapley_values(path: str):
    results = read_results(path)

    table = sorted(
        [(comb, loss, accuracy) for comb, (loss, accuracy) in results.items()],
        key=lambda x: x[2],
        reverse=True,
    )
    logging.info("\n" + tabulate(table, headers=["Combination", "Loss", "Accuracy"]))

    shapley_values = defaultdict(float)
    N = len(concepts)
    for concept in concepts:
        other_concepts = concepts.copy()
        other_concepts.remove(concept)
        for comb in get_combinations(other_concepts):
            comb_u_concept = tuple(sorted(comb + [concept]))
            comb = tuple(comb)
            coalition_size = len(comb)
            factorial_term = (
                math.factorial(coalition_size)
                * math.factorial(N - coalition_size - 1)
                / math.factorial(N)
            )

            accuracy = results[comb_u_concept][1]
            comb_accuracy = results[comb][1]

            marginal_contribution = accuracy - comb_accuracy
            shapley_values[concept] += factorial_term * marginal_contribution
    return shapley_values


def read_results(path: str) -> Dict:
    with open(path, "r") as f:
        results = json.load(f)
        results = {ast.literal_eval(key): value for key, value in results.items()}
    return results


def write_results(results: Dict, path: str):
    with open(path, "w") as f:
        results = {str(key): value for key, value in results.items()}
        json.dump(results, f)


def compute_accuracy(
    observation_shape: int,
    action_shape: int,
    activations: Dict[str, Dict],
    observations: Observation,
    probes: Dict[str, LogisticRegression],
    layer_idx: int,
):
    hidden_units = 500

    epochs = 3
    batch_size = 64
    learning_rate = 0.001
    val_split = 0.2
    test_split = 0.1

    concept_scores = np.array(
        get_concept_score(
            activations, probes, layer_idx, concept_score_method="binary"
        ),
        dtype=np.float32,
    )
    labels = np.array(observations[..., Observation.LABEL], dtype=np.float32)

    model = FeedForwardNetwork(observation_shape, hidden_units, action_shape)
    dataset = TensorDataset(
        torch.from_numpy(concept_scores), torch.tensor(labels, dtype=torch.long)
    )

    loss, accuracy = train_model(
        model=model,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        dataset=dataset,
        test_split=test_split,
        val_split=val_split,
    )

    return loss, accuracy


def get_concept_score(
    activations: Dict[str, Dict],
    probes: Dict[str, LogisticRegression],
    layer_idx: int,
    concept_score_method: Literal["binary", "soft"] = "binary",
):
    layer_activations = list(activations["latest"].values())[layer_idx]

    concept_scores = []
    for concept, probe in probes.items():
        if concept_score_method == "binary":
            concept_score = individual_binary_concept_score(layer_activations, probe)
        elif concept_score_method == "soft":
            concept_score = individual_soft_concept_score(layer_activations, probe)
        else:
            raise ValueError(
                f"Concept score method {concept_score_method} not recognized."
            )
        concept_scores.append(concept_score)

    concept_scores = list(zip(*concept_scores))

    return concept_scores


def get_probe(concept: str, layer_idx: int):
    ignore = []

    model = ModelLoader.load_latest_model_from_path("artifacts", dqn.model)
    models = {"latest": model}
    positive_observation, test_observation = load_and_split_observation(concept, 0.8)
    negative_observation, _ = load_and_split_observation("negative_" + concept, 0.8)

    test_observation_zipped = zip_observation_data(test_observation)

    test_activations, test_input, test_output = compute_activations_from_models(
        models, test_observation_zipped, ignore
    )

    probes, positive_activations, negative_activations = get_probes(
        models, positive_observation, negative_observation, ignore
    )

    layer_activations = list(test_activations["latest"].values())[layer_idx]
    probe = list(probes["latest"].values())[layer_idx]

    return probe


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
        # "random",
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

    layers = 0
    for i in count():
        try:
            run(concepts, dqn, layer_idx=i)
        except Exception as e:
            logging.warning(e)
            break
        layers = i

    for i in range(layers):
        path = os.path.join(
            "assets", "results", f"concept_combination_accuracies_{i}.json"
        )
        values = calculate_shapley_values(path)

        values = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))

        table = [(key, value) for key, value in values.items()]
        logging.info(tabulate(table, headers=["Key", "Value"]))
