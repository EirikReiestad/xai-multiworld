import logging
import os

from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.common.observation import filter_observations
from utils.core.model_loader import ModelLoader
from xailib.common.completeness_score import get_completeness_score
from xailib.utils.observation import get_observations
from xailib.utils.probes import get_probes_and_activations

logging.basicConfig(level=logging.INFO)


def main():
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
    ignore_layers = ["_fc0"]
    layer_idx = -1
    model_type = "dqn"
    artifact_path = os.path.join("artifacts")
    epochs = 20

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=True,
        artifact_path=artifact_path,
    )
    latest_model = {"model": list(models.values())[-1]}
    (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    ) = get_observations(concepts=concepts)
    observations = collect_rollouts(
        env=environment,
        artifact=artifact,
        n=1000,
        method="random",
        force_update=False,
    )
    observations = filter_observations(observations)
    logging.info("Getting probes and activations...")
    probes, positive_activations, negative_activations = get_probes_and_activations(
        concepts,
        ignore_layers,
        latest_model,
        positive_observations,
        negative_observations,
    )
    logging.info("Calculating completeness score...")
    model = list(latest_model.values())[-1]
    completeness_score = get_completeness_score(
        probes=probes,
        concepts=concepts,
        model=model,
        observations=observations,
        method="decisiontree",
        concept_score_method="binary",
        layer_idx=layer_idx,
        epochs=epochs,
        ignore_layers=ignore_layers,
        verbose=False,
    )


if __name__ == "__main__":
    main()
