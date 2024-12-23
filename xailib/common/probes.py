from utils.common.model_artifact import ModelArtifact
from utils.common.observation import Observation
from xailib.core.linear_probing.linear_probe import LinearProbe
from typing import Dict
from sklearn.linear_model import LogisticRegression


def get_probes(
    model_artifacts: dict[str, ModelArtifact],
    positive_observation: Observation,
    negative_observation: Observation,
) -> Dict[str, Dict[str, LogisticRegression]]:
    regressors = {}

    positive_observation[..., Observation.LABEL] = 1
    negative_observation[..., Observation.LABEL] = 0

    for model_name, model_artifact in model_artifacts.items():
        linear_probe = LinearProbe(
            model_artifact.model,
            positive_observation,
            negative_observation,
        )
        model_regressors = linear_probe.train()
        regressors[model_name] = model_regressors

    return regressors
