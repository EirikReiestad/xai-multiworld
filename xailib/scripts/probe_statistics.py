import logging

from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.core.model_loader import ModelLoader
from xailib.utils.metrics import calculate_statistics
from xailib.utils.observation import get_observations
from xailib.utils.probes import get_probes_and_activations


def main():
    artifact_path = "artifacts"
    model_type = "dqn"
    eval = True
    layer_idx = 4
    concepts = [
        "random",
        "goal_in_front",
        "goal_in_view",
        "goal_to_left",
        "goal_to_right",
        "wall_in_view",
    ]
    ignore_layers = ["_fc0"]

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=eval,
        artifact_path=artifact_path,
    )
    (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    ) = get_observations(concepts)
    logging.info("Getting probes and activations...")
    probes, positive_activations, negative_activations = get_probes_and_activations(
        concepts=concepts,
        models=models,
        positive_observations=positive_observations,
        negative_observations=negative_observations,
        ignore_layers=ignore_layers,
    )
    logging.info("Calculating statistics...")
    calculate_statistics(concepts, positive_activations, probes, layer_idx)


if __name__ == "__main__":
    main()
