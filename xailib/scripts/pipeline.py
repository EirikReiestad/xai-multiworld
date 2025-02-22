import json
import logging

from utils.core.model_loader import ModelLoader
from xailib.core.pipeline.collect_rollouts import collect_rollouts
from xailib.core.pipeline.generate_concepts import generate_concepts
from xailib.utils.pipeline_utils import (
    create_environment,
    download_models,
    get_activations,
    get_concept_scores,
    get_models,
    get_observations,
    get_probes_and_activations,
    get_tcav_scores,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    with open("xailib/configs/pipeline_config.json", "r") as f:
        config = json.load(f)

    download_models(config)

    artifact = ModelLoader.load_latest_model_artifacts_from_path(
        config["path"]["artifacts"]
    )
    logging.info(f"Loaded model artifact metadata: {artifact.metadata}")

    environment = create_environment(artifact)
    collect_rollouts(config, environment, artifact)
    generate_concepts(config, environment, artifact)

    (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    ) = get_observations(config)

    models = get_models(config, environment, artifact)

    probes, positive_activations, negative_activations = get_probes_and_activations(
        config, models, positive_observations, negative_observations
    )

    test_positive_activations, test_input, test_output = get_activations(
        config, test_positive_observations, models
    )

    concept_scores = get_concept_scores(config, test_positive_activations, probes)
    tcav_scores = get_tcav_scores(
        config, test_positive_activations, test_output, probes
    )


if __name__ == "__main__":
    main()
