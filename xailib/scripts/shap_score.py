import logging
import os

from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.core.model_loader import ModelLoader
from xailib.core.shap.calculate_shap import calculate_shap

logging.basicConfig(level=logging.INFO)


def main():
    model_type = "dqn"
    eval = True
    artifact_path = os.path.join("artifacts")

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=eval,
        artifact_path=artifact_path,
    )
    model = list(models.values())[-1]

    observations = collect_rollouts(
        env=environment,
        artifact=artifact,
        n=10,
        method="policy",
        force_update=True,
    )

    calculate_shap(artifact, environment, model, observations)


if __name__ == "__main__":
    main()
