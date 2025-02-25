from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.core.model_loader import ModelLoader
from xailib.utils.activations import get_concept_activations
from xailib.utils.metrics import get_tcav_scores
from xailib.utils.observation import get_observations
from xailib.utils.probes import get_probes_and_activations


def main():
    artifact_path = "artifacts"
    model_type = "dqn"
    eval = True
    concepts = [
        "random",
        "goal_in_front",
        # "goal_in_view",
        # "goal_to_left",
        # "goal_to_right",
        # "wall_in_view",
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
    test_positive_activations, test_input, test_output = get_concept_activations(
        concepts, test_positive_observations, models, ignore_layers
    )
    probes, positive_activations, negative_activations = get_probes_and_activations(
        concepts=concepts,
        ignore_layers=ignore_layers,
        models=models,
        positive_observations=positive_observations,
        negative_observations=negative_observations,
    )
    tcav_scores = get_tcav_scores(
        concepts, test_positive_activations, test_output, probes, show=True
    )


if __name__ == "__main__":
    main()
