import json
import logging

from utils.core.model_loader import ModelLoader
from xailib.core.pipeline.collect_rollouts import collect_rollouts, generate_concepts
from xailib.utils.pipeline_utils import (
    create_environment,
    download_models,
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

    environment = create_environment(config, artifact)
    collect_rollouts(config, environment, artifact)
    generate_concepts(config, environment, artifact)


if __name__ == "__main__":
    main()
