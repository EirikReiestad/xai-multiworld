from typing import Dict
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from xailib.common.activations import preprocess_activations


def binary_concept_score(activations: np.ndarray, probe: LogisticRegression) -> float:
    labels = np.ones(activations.shape[0])
    score = 2 * max(probe.score(activations, labels) - 0.5, 0)
    return score


def binary_concept_scores(
    activations: Dict[str, Dict[str, Dict]],  # activations[model_name][layer_name]
    probes: Dict[str, Dict[str, LogisticRegression]],  # probes[model_name][layer_name]
) -> Dict[str, Dict[str, float]]:
    assert (
        activations.keys() == probes.keys()
    ), "Activations and probes must have the same keys"

    for model_name in activations.keys():
        assert (
            len(activations[model_name].keys()) == len(probes[model_name].keys())
        ), f"Activations and probes must have the number of keys, got {activations[model_name].keys()} and {probes[model_name].keys()}"

    scores = defaultdict(dict)
    for model_name, model in probes.items():
        for (layer_name, probe), layer_activation in zip(
            model.items(), activations[model_name].values()
        ):
            preprocessed_activations = preprocess_activations(layer_activation)
            scores[model_name][layer_name] = binary_concept_score(
                preprocessed_activations, probe
            )
    return scores
