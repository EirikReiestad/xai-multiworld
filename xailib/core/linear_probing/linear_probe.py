from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from utils.common.observation import (
    Observation,
    zip_observation_data,
)
from xailib.common.activations import ActivationTracker, preprocess_activations


class LinearProbe:
    def __init__(
        self,
        model: nn.Module,
        positive_observations: Observation,
        negative_observations: Observation,
        ignore: List[str] = [],
    ):
        self._model = model
        self._model.eval()

        self._positive_observations = positive_observations
        self._negative_observations = negative_observations

        self._activation_tracker = ActivationTracker(self._model, ignore)

        np.random.seed(None)

    def train(self) -> Tuple[Dict[str, LogisticRegression], Dict, Dict]:
        positive_observations, _ = zip_observation_data(self._positive_observations)
        negative_observations, _ = zip_observation_data(self._negative_observations)

        positive_activations, positive_input, positive_output = (
            self._activation_tracker.compute_activations(positive_observations)
        )
        negative_activations, negative_input, negative_output = (
            self._activation_tracker.compute_activations(negative_observations)
        )
        self._activation_tracker.clean()

        regressors = {}

        assert positive_activations.keys() == negative_activations.keys(), (
            f"Positive and negative activations must have the same layers. "
            f"Positive: {positive_activations.keys()}, Negative: {negative_activations.keys()}"
        )
        assert len(positive_activations.keys()) > 0, "No activations found"

        for i, layer in enumerate(positive_activations.keys()):
            # name = f"{i}_{layer.__class__.__name__}"
            name = layer
            regressor = LinearProbe.compute_regressor(
                positive_activations[layer], negative_activations[layer]
            )
            regressors[name] = regressor

        return regressors, positive_activations, negative_activations

    @staticmethod
    def compute_regressor(
        positive_activations: dict,
        negative_activations: dict,
    ) -> LogisticRegression:
        pos_act = preprocess_activations(positive_activations)
        neg_act = preprocess_activations(negative_activations)

        assert pos_act.shape[1] == neg_act.shape[1]

        positive_labels = np.ones(pos_act.shape[0])
        negative_labels = np.zeros(neg_act.shape[0])

        combined_activations = np.concatenate([pos_act, neg_act])
        combined_labels = np.concatenate([positive_labels, negative_labels])

        idx = np.random.permutation(combined_activations.shape[0])
        combined_activations = combined_activations[idx]
        combined_labels = combined_labels[idx]

        regressor = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
        regressor.fit(combined_activations, combined_labels)

        return regressor

    def _tcav_score(
        self, activations: dict, network_output: torch.Tensor, cav: np.ndarray
    ) -> float:
        torch_activations = activations["output"]
        assert isinstance(
            torch_activations, torch.Tensor
        ), "Activations must be a tensor"
        assert (
            torch_activations.requires_grad
        ), "Activations must have requires_grad=True"
        sensitivity_score = self._sensitivity_score(
            torch_activations, network_output, cav
        )
        return (sensitivity_score > 0).mean()

    def _sensitivity_score(
        self, activations: torch.Tensor, network_output: torch.Tensor, cav: np.ndarray
    ) -> np.ndarray:
        assert isinstance(activations, torch.Tensor), "Activations must be a tensor"
        assert activations.requires_grad, "Activations must have requires_grad=True"
        assert cav.ndim == 2, "Coef must be 2D (n_features, n_classes)"

        output = network_output.max(dim=1)[0]
        grads = torch.autograd.grad(
            output,
            activations,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
        )[0]

        grads_flattened = grads.view(grads.size(0), -1).detach().numpy()
        sensitivity_score = np.dot(grads_flattened, cav.T)
        return sensitivity_score

    def _binary_concept_score(
        self, activations: dict, regressor: LogisticRegression
    ) -> float:
        act = preprocess_activations(activations)
        labels = np.ones(act.shape[0])
        score = 2 * max(regressor.score(act, labels) - 0.5, 0)
        return score
