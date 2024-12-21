import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning

# from rl.src.regressor.logistic import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.common.observation import Observation

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class LinearProbe:
    id: int = 0

    def __init__(
        self,
        model: nn.Module,
        positive_observations: List[Observation],
        negative_observations: List[Observation],
        test_positive_observations: List[Observation],
        scaler: str = "",
    ):
        self._model = model
        self._model.eval()

        self._positive_observations = positive_observations
        self._negative_observations = negative_observations
        self._test_positive_observations = test_positive_observations

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

    def _register_hooks(self):
        for name, layer in self._model.named_children():
            if not isinstance(layer, nn.Sequential):
                continue
            for sub_layer in layer:
                if not isinstance(sub_layer, nn.ReLU):
                    continue
                sub_layer.register_forward_hook(self._module_hook)

    def compute_cavs(self) -> tuple[dict, dict, dict]:
        positive_data, positive_labels = self._positive_data.get_data_lists(
            sample_ratio
        )
        negative_data, negative_labels = self._negative_data.get_data_lists(
            sample_ratio
        )
        test_data, test_labels = self._test_positive_data.get_data_lists(sample_ratio)

        if custom_test_data is not None:
            test_data_handler = DataHandler()
            test_data_handler.load_samples(custom_test_data)
            test_data, test_labels = test_data_handler.get_data_lists(sample_ratio=1.0)

        positive_activations, positive_output = self._compute_activations(
            positive_data, requires_grad=True
        )
        negative_activations, negative_output = self._compute_activations(
            negative_data, requires_grad=True
        )
        test_activations, test_output = self._compute_activations(
            test_data, requires_grad=True
        )

        cavs = {}
        binary_concept_scores = {}
        tcav_scores = {}

        for i, layer in enumerate(self._activations.keys()):
            self._layer_idx = i
            if self._scaler_name == "minmax":
                self._scaler = MinMaxScaler()
            elif self._scaler_name == "standard":
                self._scaler = StandardScaler()
            name = f"{i}_{layer.__class__.__name__}"

            regressor = self._compute_regressor(
                positive_activations[layer], negative_activations[layer]
            )
            cav = self._cav(regressor)
            self._plot_distribution(cav.flatten(), name + "_cav" + str(CAV.id))
            tcav_score = self._tcav_score(test_activations[layer], test_output, cav)
            binary_concept_score = self._binary_concept_score(
                test_activations[layer], regressor
            )

            cavs[name] = cav
            binary_concept_scores[name] = binary_concept_score
            tcav_scores[name] = tcav_score

            if plot_distribution is True:
                self._plot_distribution(
                    positive_activations[layer]["output"].detach().numpy(),
                    name,
                )
                self._plot_distribution(
                    negative_activations[layer]["output"].detach().numpy(),
                    name + "_negative",
                )

        CAV.id += 1
        return cavs, binary_concept_scores, tcav_scores

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

        # Initialize weights randomly
        regressor = LogisticRegression(max_iter=200, solver="lbfgs", C=1.0)
        """
        n_features = combined_activations.shape[1]
        random_state = np.random.RandomState(seed=None)
        initial_weights = random_state.normal(size=n_features)

        regressor = LogisticRegression(
            max_iter=200, solver="lbfgs", warm_start=True, random_state=random_state
        )

        regressor.coef_ = np.array([initial_weights])
        regressor.intercept_ = np.array([random_state.normal()])
        """

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
        self._plot_distribution(
            sensitivity_score, f"sensitivity_score_{self._layer_idx}_{CAV.id}"
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
        self._plot_distribution(
            grads_flattened,
            f"grads_flattened_{self._layer_idx}_{CAV.id}",
        )
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
        self, inputs: list[np.ndarray], requires_grad=False
    ) -> tuple[dict, torch.Tensor]:
        self._activations.clear()

        torch_inputs: list[torch.Tensor] = [
            torch.from_numpy(input).requires_grad_(True) for input in inputs
        ]

        inputs = torch.stack(torch_inputs).requires_grad_(True)

        outputs = self._model(inputs)

        activations_cloned = {key: value for key, value in self._activations.items()}
        return activations_cloned, outputs

    def _module_hook(self, module: nn.Module, input, output):
        self._activations[module] = {
            "input": input[0],
            "output": output,
        }
