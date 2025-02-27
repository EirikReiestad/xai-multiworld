import os

from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.core.model_loader import ModelLoader
from xailib.common.completeness_score import get_completeness_score
from xailib.utils.observation import get_observations
from xailib.utils.probes import get_probes_and_activations


def main():
    concepts = [
        "random",
        "goal_in_front",
        "goal_in_view",
        "goal_to_left",
        "goal_to_right",
        "wall_in_view",
    ]
    ignore_layers = ["_fc0"]
    layer_idx = 4
    model_type = "dqn"
    artifact_path = os.path.join("artifacts")

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=True,
        artifact_path=artifact_path,
    )
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
        method="policy",
    )
    probes, positive_activations, negative_activations = get_probes_and_activations(
        concepts, ignore_layers, models, positive_observations, negative_observations
    )
    completeness_score = get_completeness_score(
        probes,
        concepts,
        artifact,
        environment,
        observations,
        layer_idx,
        ignore_layers=ignore_layers,
    )


if __name__ == "__main__":
    main()
