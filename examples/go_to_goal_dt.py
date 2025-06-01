import logging
import os
from functools import partial
from itertools import count

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data.dataset import TensorDataset

from multiworld.multigrid.core.concept import get_concept_check_bitmap
from multiworld.multigrid.utils.decoder import decode_observation
from multiworld.multigrid.utils.ohe import decode_direction
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.observation import Observation, filter_observations
from utils.core.model_loader import ModelLoader
from xailib.common.generate_concepts import generate_concepts
from xailib.common.train_model import train_decision_tree

logging.basicConfig(level=logging.INFO)


def main(
    n: int = 1000,
    force_update: bool = False,
    filename: str = "decision_tree_human.json",
):
    concepts = [
        "random",
        "goal_in_front",
        "goal_in_view",
        "goal_to_left",
        "goal_to_right",
        "goal_at_top",
        "goal_at_bottom",
        "next_to_goal",
        "agent_in_view",
        "agent_to_right",
        "agent_to_left",
        "agent_in_front",
        "rotated_left",
        "rotated_right",
        "rotated_up",
        "rotated_down",
        "wall_in_view",
        "wall_in_front",
        "wall_to_right",
        "wall_to_left",
        "next_to_wall",
        "close_to_wall",
    ]
    artifact_path = os.path.join("artifacts")
    epochs = 1

    decode_obs = partial(
        decode_observation, preprocessing=PreprocessingEnum.ohe_minimal
    )

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
    observations = collect_rollouts(
        env=environment,
        artifact=artifact,
        n=n,
        sample_rate=0.1,
        method="policy",
        force_update=force_update,
    )
    observations = filter_observations(observations)

    test_split = 0.2

    concept_scores = []
    for obs in observations[..., Observation.OBSERVATION]:
        grid = np.array(obs[0]["observation"])
        grid = decode_obs({"observation": grid})
        direction = decode_direction(obs[0]["direction"])
        state = {"observation": grid["observation"], "direction": direction}
        concept_score = get_concept_check_bitmap(state, concepts)
        concept_scores.append(concept_score)
    concept_scores = np.array(concept_scores)
    labels = np.array(observations[..., Observation.LABEL], dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(concept_scores), torch.tensor(labels, dtype=torch.long)
    )
    model = DecisionTreeClassifier()

    model = train_decision_tree(
        model=model,
        dataset=dataset,
        test_split=test_split,
        feature_names=concepts.copy(),
        epochs=epochs,
        filename=filename,
    )

    render = False
    if not render:
        return

    environment = create_environment(artifact, render_mode="human")
    while True:
        observations, _ = environment.reset()
        for i in count():
            actions = {}
            for key, observation in observations.items():
                grid = observation["observation"]
                obs = decode_obs({"observation": grid})
                grid = obs["observation"]
                direction = decode_direction(observation["direction"])
                state = {"observation": grid, "direction": direction}
                concept_score = get_concept_check_bitmap(state, concepts)
                concept_score = np.array(concept_score).reshape(1, -1)
                action = model.predict(concept_score)
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
    for i in range(10):
        main(filename=f"decision_tree_human_{i}.json")
