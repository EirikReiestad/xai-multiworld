from typing import Dict, List, Tuple

import numpy as np
import torch.nn as nn

from utils.common.observation import Observation, zip_observation_data
from xailib.common.activations import compute_activations_from_models


def get_activations(
    models: Dict[str, nn.Module], observations: Observation, ignore_layers: List = []
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
]:
    observation_zipped = zip_observation_data(observations)
    activations, input, output = compute_activations_from_models(
        models, observation_zipped, ignore_layers
    )
    return activations, input, output


def get_concept_activations(
    concepts: List[str],
    observation: Dict[str, Observation],
    models: Dict[str, nn.Module],
    ignore_layers: List = [],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
]:
    activations = {}
    inputs = {}
    outputs = {}

    for concept in concepts:
        observation_zipped = zip_observation_data(observation[concept])
        activation, input, output = compute_activations_from_models(
            models, observation_zipped, ignore_layers
        )
        activations[concept] = activation
        inputs[concept] = input
        outputs[concept] = output
    return activations, inputs, outputs
