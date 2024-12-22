from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.common.observation import Observation


def binary_concept_score(activations: Any, probe: LogisticRegression) -> float:
    labels = np.ones(activations.shape[0])
    score = 2 * max(probe.score(activations, labels) - 0.5, 0)
    return score
