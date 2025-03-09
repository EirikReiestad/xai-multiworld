from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from xailib.common.activations import preprocess_activations


def binary_concept_score(activations: np.ndarray, probe: LogisticRegression) -> float:
    labels = np.ones(activations.shape[0])
    score = 2 * max(probe.score(activations, labels) - 0.5, 0)
    return score


def soft_concept_score(activations: np.ndarray, probe: LogisticRegression) -> float:
    # probe_probs = probe.predict_proba(activations)[:, 1]

    activations = activations.reshape(1, -1)
    coef = probe.coef_
    coef = coef.reshape(1, -1)

    similarity = cosine_similarity(activations, coef)
    return similarity[0][0]


def individual_binary_concept_score(
    activations: Dict, probe: LogisticRegression
) -> List[float]:
    scores = []
    preprocessed_activations = preprocess_activations(activations)
    for activation in preprocessed_activations:
        score = binary_concept_score(np.array([activation]), probe)
        scores.append(score)
    return scores


def individual_soft_concept_score(
    activations: Dict, probe: LogisticRegression
) -> List[float]:
    scores = []
    preprocessed_activations = preprocess_activations(activations)
    for activation in preprocessed_activations:
        score = soft_concept_score(np.array([activation]), probe)
        scores.append(score)
    return scores


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
        ), f"Activations and probes must have the number of keys, got {len(activations[model_name].keys())} and {len(probes[model_name].keys())}"

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
