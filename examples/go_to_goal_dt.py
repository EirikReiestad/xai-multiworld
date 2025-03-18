import logging
import os
from functools import partial
from itertools import count

import numpy as np

from multiworld.multigrid.core.concept import get_concept_check_bitmap
from multiworld.multigrid.utils.decoder import decode_observation
from multiworld.multigrid.utils.ohe import decode_direction
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.common.observation import filter_observations
from utils.core.model_loader import ModelLoader
from xailib.common.completeness_score import get_completeness_score
from xailib.common.generate_concepts import generate_concepts
from xailib.utils.observation import get_observations
from xailib.utils.probes import get_probes_and_activations

logging.basicConfig(level=logging.INFO)


def main():
    concepts = [
        "random",
        "goal_in_view",
        "goal_to_right",
        "goal_to_left",
        "goal_in_front",
        "goal_at_top",
        "goal_at_bottom",
        "goal_around_middle_front",
        "next_to_goal",
        # "agent_in_view",
        # "agent_to_right",
        # "agent_to_left",
        # "agent_in_front",
        "rotated_right",
        "rotated_left",
        "rotated_up",
        "rotated_down",
        "wall_in_view",
        "wall_in_front",
        "wall_to_right",
        "wall_to_left",
        "next_to_wall",
        "close_to_wall",
    ]
    ignore_layers = ["_fc0"]
    layer_idx = -1
    model_type = "dqn"
    artifact_path = os.path.join("artifacts")
    epochs = 1

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact, width=10, height=10, static=True)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=True,
        artifact_path=artifact_path,
    )
    latest_model = {"model": list(models.values())[-1]}
    generate_concepts(
        concepts=concepts,
        env=environment,
        observations=1000,
        artifact=artifact,
        method="policy",
        force_update=False,
    )
    (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    ) = get_observations(concepts=concepts.copy())
    observations = collect_rollouts(
        env=environment,
        artifact=artifact,
        n=1000,
        sample_rate=1,
        method="policy",
        force_update=True,
    )
    observations = filter_observations(observations)
    logging.info("Getting probes and activations...")
    probes, positive_activations, negative_activations = get_probes_and_activations(
        concepts.copy(),
        ignore_layers,
        latest_model,
        positive_observations,
        negative_observations,
    )
    logging.info("Calculating completeness score...")
    model = list(latest_model.values())[-1]
    model = get_completeness_score(
        probes=probes,
        concepts=concepts.copy(),
        model=model,
        observations=observations,
        method="decisiontree",
        concept_score_method="binary",
        layer_idx=layer_idx,
        epochs=epochs,
        ignore_layers=ignore_layers,
        verbose=False,
    )

    environment = create_environment(
        artifact, width=10, height=10, static=True, render_mode="human"
    )
    while True:
        observations, _ = environment.reset()
        for i in count():
            actions = {}
            for key, observation in observations.items():
                grid = observation["observation"]
                decode_obs = partial(
                    decode_observation, preprocessing=PreprocessingEnum.ohe_minimal
                )
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
    main()
