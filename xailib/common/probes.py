from typing import Dict, List

import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from utils.common.observation import Observation
from xailib.core.linear_probing.linear_probe import LinearProbe


def get_probes(
    models: Dict[str, nn.Module],
    positive_observation: Observation,
    negative_observation: Observation,
    ignore: List["str"] = [],
) -> Dict[str, Dict[str, LogisticRegression]]:
    regressors = {}

    positive_observation[..., Observation.LABEL] = 1
    negative_observation[..., Observation.LABEL] = 0

    for model_name, model in models.items():
        linear_probe = LinearProbe(
            model,
            positive_observation,
            negative_observation,
            ignore,
        )
        model_regressors = linear_probe.train()
        regressors[model_name] = model_regressors

    return regressors
