from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.core.model_loader import ModelLoader
from xailib.common.generate_concepts import generate_concepts
from xailib.utils.activations import get_concept_activations
from xailib.utils.metrics import get_concept_scores
from xailib.utils.observation import get_observations
from xailib.utils.probes import get_probes_and_activations


def main():
    artifact_path = "artifacts"
    model_type = "dqn"
    eval = True
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

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=eval,
        artifact_path=artifact_path,
    )
    generate_concepts(
        concepts=concepts,
        env=environment,
        observations=100,
        artifact=artifact,
        method="policy",
        force_update=True,
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
    concept_scores = get_concept_scores(
        concepts, test_positive_activations, probes, show=True
    )


if __name__ == "__main__":
    main()
