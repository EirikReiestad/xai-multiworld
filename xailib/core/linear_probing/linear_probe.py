import warnings
from typing import Any, List, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning

# from rl.src.regressor.logistic import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.common.observation import (
    Observation,
    split_observation,
    zip_observation_data,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class LinearProbe:
    def __init__(
        self,
        model: nn.Module,
        positive_observations: Observation,
        negative_observations: Observation,
        scaler: str = "",
    ):
        self._model = model
        self._model.eval()

        self._positive_observations, self._test_positive_observations = (
            split_observation(positive_observations, 0.8)
        )
        self._negative_observations, _ = split_observation(negative_observations, 0.8)

        self._register_hooks()
        self._activations = {}

        self._scaler_name = scaler  # "minmax", "standard", or ""
        assert self._scaler_name in [
            "minmax",
            "standard",
            "",
        ], "Invalid scaler name"
        self._scaler = None

        np.random.seed(None)

    # TODO: Make this modular and allow for different networks
    def _register_hooks(self):
        for _, processor in self._model.named_children():
            for _, layer in processor.named_children():
                for name, sub_layer in layer.named_children():
                    if not isinstance(sub_layer, nn.ReLU):
                        continue
                    sub_layer.register_forward_hook(self._module_hook)

    def train(self) -> Dict[str, LogisticRegression]:
        positive_observations = zip_observation_data(self._positive_observations)
        negative_observations = zip_observation_data(self._negative_observations)

        positive_activations, positive_output = self._compute_activations(
            positive_observations, requires_grad=True
        )
        negative_activations, negative_output = self._compute_activations(
            negative_observations, requires_grad=True
        )

        regressors = {}

        for i, layer in enumerate(self._activations.keys()):
            name = f"{i}_{layer.__class__.__name__}"
            regressor = self._compute_regressor(
                positive_activations[layer], negative_activations[layer]
            )
            regressors[name] = regressor

        return regressors

    def _preprocess_activations(self, activations: dict) -> np.ndarray:
        numpy_activations = activations["output"].detach().numpy()
        reshaped_activations = numpy_activations.reshape(numpy_activations.shape[0], -1)
        if self._scaler is not None:
            scaled_act = self._scaler.fit_transform(reshaped_activations)
            return scaled_act
        return reshaped_activations

    def _plot_distribution(self, activations: np.ndarray, filename: str):
        filename += self._scaler_name
        filepath = f"assets/figures/act_dist{filename}.png"
        flatten_activations = activations.flatten()
        fig, ax = plt.subplots()
        assert (
            len(flatten_activations) == activations.size
        ), f"Activations must be flattened, shape: {activations.shape}, flatten shape: {flatten_activations.shape}"
        assert isinstance(
            flatten_activations[0], (float, np.floating)
        ), f"Activations must be float, not {type(flatten_activations[0])}"
        try:
            sns.histplot(flatten_activations, kde=True, stat="density", ax=ax)
        except ValueError as e:
            raise ValueError(
                f"Error while plotting distribution for layer {filename}, activations shape: {flatten_activations.shape}, activations: {flatten_activations}"
            ) from e
        plt.xlabel("Activation")
        plt.ylabel("Density")
        plt.title(f"Activation Distribution for layer {filename}")
        plt.savefig(filepath)

    def _cav(self, regressor: LogisticRegression):
        return regressor.coef_

    def _compute_regressor(
        self,
        positive_activations: dict,
        negative_activations: dict,
    ) -> LogisticRegression:
        pos_act = self._preprocess_activations(positive_activations)
        neg_act = self._preprocess_activations(negative_activations)

        assert pos_act.shape[1] == neg_act.shape[1]

        positive_labels = np.ones(pos_act.shape[0])
        negative_labels = np.zeros(neg_act.shape[0])

        combined_activations = np.concatenate([pos_act, neg_act])
        combined_labels = np.concatenate([positive_labels, negative_labels])

        idx = np.random.permutation(combined_activations.shape[0])
        combined_activations = combined_activations[idx]
        combined_labels = combined_labels[idx]

        regressor = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
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
        act = self._preprocess_activations(activations)
        labels = np.ones(act.shape[0])
        score = 2 * max(regressor.score(act, labels) - 0.5, 0)
        return score

    def _compute_activations(
        self, inputs: List, requires_grad=True
    ) -> tuple[dict, torch.Tensor]:
        self._activations.clear()

        outputs = self._model(*inputs)

        activations_cloned = {key: value for key, value in self._activations.items()}
        return activations_cloned, outputs

    def _module_hook(self, module: nn.Module, input, output):
        self._activations[module] = {
            "input": input[0],
            "output": output,
        }
