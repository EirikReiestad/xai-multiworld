from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch import nn

from utils.common.observation import Observation
from xailib.common.probes import get_probes


def get_probes_and_activations(
    concepts: List[str],
    ignore_layers: List[str],
    models: Dict[str, nn.Module],
    positive_observations: Dict[str, Observation],
    negative_observations: Dict[str, Observation],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, LogisticRegression]]],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    probes = {}
    positive_activations = {}
    negative_activations = {}

    for concept in concepts:
        positive_observation = positive_observations[concept]
        negative_observation = negative_observations[concept]
        probe, positive_activation, negative_activation = get_probes(
            models, positive_observation, negative_observation, ignore_layers
        )
        probes[concept] = probe
        positive_activations[concept] = positive_activation
        negative_activations[concept] = negative_activation

    return probes, positive_activations, negative_activations
