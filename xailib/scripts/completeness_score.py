from typing import Dict

from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.core.model_loader import ModelLoader
from xailib.common.completeness_score import get_completeness_score
from xailib.utils.observation import get_observations
from xailib.utils.probes import get_probes_and_activations


def main(config: Dict):
    concepts = config["concepts"]
    ignore_layers = config["analyze"]["ignore_layers"]
    layer_idx = config["analyze"]["layer_idx"]

    artifact = ModelLoader.load_latest_model_artifacts_from_path(
        config["path"]["artifacts"]
    )
    environment = create_environment(artifact)
    models = get_models(
        artifact,
        config["model"]["type"],
        config["path"]["artifacts"],
        environment,
        eval=True,
    )
    (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    ) = get_observations(config)
    observations = collect_rollouts(environment, artifact)
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
        model_type=config["model"]["type"],
    )


if __name__ == "__main__":
    config = {
        "analyze": {"ignore_layers": ["_fc0"], "layer_idx": 4},
        "model": {"type": "dqn"},
        "concepts": [
            "random",
            "goal_in_front",
            "goal_in_view",
            "goal_to_left",
            "goal_to_right",
            "wall_in_view",
        ],
        "path": {"artifacts": "artifacts"},
    }
    main(config)
