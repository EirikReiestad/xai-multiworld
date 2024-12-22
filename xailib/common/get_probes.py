import os

import torch.nn as nn
from torch import positive

from utils.common.model_artifact import ModelArtifact
from utils.common.observation import Observation, observation_from_file
from xailib.core.linear_probing.linear_probe import LinearProbe


def get_probes(
    model_artifacts: dict[str, ModelArtifact],
    positive_observation: Observation,
    negative_observation: Observation,
    concept_path: str = os.path.join("assets", "concepts"),
):
    regressors = {}

    positive_observation[..., Observation.LABEL] = 1
    negative_observation[..., Observation.LABEL] = 0

    for model_name, model_artifact in model_artifacts.items():
        linear_probe = LinearProbe(
            model_artifact.model,
            positive_observation,
            negative_observation,
        )
        regressors = linear_probe.train()

    return regressors
