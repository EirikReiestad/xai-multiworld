import logging
import os
from collections import defaultdict
from itertools import count
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data.dataset import TensorDataset

from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.common.observation import Observation, filter_observations
from utils.core.model_loader import ModelLoader
from xailib.common.train_model import train_decision_tree
from xailib.core.calculate_cavs.calculate_cavs import calculate_cavs

logging.basicConfig(level=logging.INFO)


def main(
    n: int = 1000, force_update: bool = False, filename: str = "decision_tree.json"
):
    artifact_path = "artifacts"
    model_type = "dqn"
    eval = True
    ignore_layers = ["_fc0"]
    result_path = os.path.join("assets", "results")
    method = "policy"

    M = 15
    lambda_1 = 1 / 3
    lambda_2 = 1 / 3
    lambda_3 = 1 / 3
    batch_size = 128
    lr = 1e-1
    epochs = 1

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=eval,
        artifact_path=artifact_path,
    )
    model = list(models.values())[-1]
    logging.info("Collecting rollouts...")
    observations = collect_rollouts(
        env=environment,
        artifact=artifact,
        n=n,
        sample_rate=0.1,
        method=method,
        force_update=force_update,
    )
    observations = filter_observations(observations)

    average_instance_of_each_class = 100
    total_number_of_instances = 1000
    average_class_ratio = average_instance_of_each_class / total_number_of_instances
    K = int(batch_size * average_class_ratio / 2)

    logging.info("Calculating CAVs...")
    (
        cavs,
        stats,
        positive_observations,
        negative_observations,
        positive_activations,
        negative_observations,
        similarity_weights,
    ) = calculate_cavs(
        model=model,
        env=environment,
        artifact=artifact,
        method="policy",
        M=M,
        K=K,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        num_observations=2000,
        num_sample_observations=200,
        ignore_layers=ignore_layers,
    )

    average_positive_observations = {}
    for key, value in positive_observations.items():
        observations_ = []
        directions = []
        for obs in value:
            observations_.append(obs["observation"])
            directions.append(obs["direction"])
        avg_observations = np.mean(observations_, axis=0)
        avg_directions = np.mean(directions, axis=0)
        observation = {"observation": avg_observations, "direction": avg_directions}
        average_positive_observations[key] = observation

    def concept_score(cav_obs: Dict[str, Dict], other_obs: List[Dict]) -> List[float]:
        score = []
        for key, obs in cav_obs.items():
            diff = obs_diff(obs, other_obs)
            score.append(diff)
        return score

    def obs_diff(obs: Dict, other_obs: List[Dict]):
        obs_diff = np.abs(obs["observation"] - [o["observation"] for o in other_obs])
        dir_diff = np.abs(obs["direction"] - [o["direction"] for o in other_obs])
        total_diff = obs_diff.mean().sum() + dir_diff.mean().sum()
        max_obs_diff = np.prod(obs["observation"].shape)
        max_dir_diff = np.prod(obs["direction"].shape)
        max_diff = max_obs_diff + max_dir_diff
        diff = total_diff / max_diff
        return diff

    diff_matrix = defaultdict(dict)
    for key, value in positive_observations.items():
        for other_key, other_value in positive_observations.items():
            diff_matrix[key][other_key] = obs_diff(
                average_positive_observations[key], other_value
            )

    df = pd.DataFrame(diff_matrix)
    logging.info(f"Similarity matrix:\n{df}")

    concept_scores = []
    for obs in observations[..., Observation.OBSERVATION]:
        grid = obs[0]["observation"]
        direction = obs[0]["direction"]
        state = {"observation": grid, "direction": direction}
        score = concept_score(average_positive_observations, [state])
        concept_scores.append(score)

    concept_scores = np.array(concept_scores)
    labels = np.array(observations[..., Observation.LABEL], dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(concept_scores), torch.tensor(labels, dtype=torch.long)
    )
    model = DecisionTreeClassifier()

    test_split = 0.2
    feature_names = [str(i) for i in range(len(cavs))]
    model = train_decision_tree(
        model=model,
        dataset=dataset,
        test_split=test_split,
        feature_names=feature_names,
        epochs=epochs,
        filename=filename,
    )

    render = False
    if not render:
        return

    environment = create_environment(artifact, static=False, render_mode="human")
    while True:
        observations, _ = environment.reset()
        for i in count():
            actions = {}
            for key, observation in observations.items():
                grid = observation["observation"]
                state = {"observation": grid, "direction": observation["direction"]}
                score = concept_score(average_positive_observations, [state])
                score = np.array(score).reshape(1, -1)
                action = model.predict(score)
                actions[key] = action
            observations, rewards, terminations, truncations, info = environment.step(
                actions
            )
            dones = {}
            for key in terminations.keys():
                dones[key] = terminations[key] or truncations[key]
            if all(dones.values()):
                break


if __name__ == "__main__":
    main(force_update=True, filename="decision_tree_0.json")
    for i in range(1, 10):
        main(filename=f"decision_tree_{i}.json")
