from typing import Dict, List, Tuple
from numpy.typing import NDArray
from collections import defaultdict
from utils.common.observation import Observation, zip_observation_data, set_require_grad

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from xailib.common.activations import preprocess_activations
from xailib.common.gradients import calculate_gradients


def feature_concept_importance(
    activations: Dict[str, Dict[str, Dict]],  # activations[model_name][layer_name]
    observations: Dict[str, List[torch.Tensor]],
    probes: Dict[str, Dict[str, LogisticRegression]],  # probes[model_name][layer_name]
) -> Dict[str, Dict[str, float]]:
    assert (
        activations.keys() == probes.keys()
    ), "Activations and probes must have the same keys"

    for model_name in activations.keys():
        assert (
            len(activations[model_name].keys()) == len(probes[model_name].keys())
        ), f"Activations and probes must have the number of keys, got {activations[model_name].keys()} and {probes[model_name].keys()}"

    # positive_activations, positive_observations = filter_negative_concepts(activations, observations, probes)

    for model_name, model in activations.items():
        obs = observations[model_name]
        for layer_name, layer_activations in model.items():
            print(model_name, layer_name)
            obs_img = obs[0].detach().clone().requires_grad_(True)
            obs_dir = obs[1].detach().clone().requires_grad_(True)
            act = layer_activations["output"]
            grads_img = calculate_gradients(obs_img, act)
            grads_dir = calculate_gradients(obs_dir.clone(), act, allow_unused=True)
            print(grads_img)
            print(grads_dir)


def filter_negative_concepts(
    activations: Dict[str, Dict[str, Dict]],  # activations[model_name][layer_name]
    observations: Observation,
    probes: Dict[str, Dict[str, LogisticRegression]],  # probes[model_name][layer_name]
) -> Tuple[Dict, Dict]:
    # TODO: Creates some problems with keeping track of the gradients and to actually find the concept mask (as everything is passed into the model at once)
    positive_activations: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    positive_observations: Dict[str, Dict[str, Observation]] = defaultdict(dict)
    for model_name, model in probes.items():
        for (layer_name, probe), layer_activation in zip(
            model.items(), activations[model_name].values()
        ):
            output_activation = layer_activation["output"]
            concept_mask = _concept_mask(output_activation, probe)

            positive_activations[model_name][layer_name] = output_activation[
                concept_mask
            ]
            positive_observations[model_name][layer_name] = observations[concept_mask]

    return positive_activations, positive_observations


def _concept_mask(target: NDArray, probe: LogisticRegression) -> NDArray:
    labels = np.ones(target.shape[0])
    scores = []
    for trg, label in zip(target, labels):
        trg = trg.reshape(1, -1)
        score = probe.score(trg, [label])
        assert score == 0 or score == 1
        scores.append(bool(score))
    return np.array(scores)
