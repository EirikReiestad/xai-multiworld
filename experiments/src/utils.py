import copy
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List

import aenum as enum
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tabulate import tabulate
from torch.utils.data import TensorDataset, random_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def log_decision_tree_feature_importance(feature_importance):
    table_data = sorted(
        feature_importance.items(), key=lambda item: item[1][0], reverse=True
    )

    formatted_data = [
        (concept, f"{value[0]:.6f}", value[1]) for concept, value in table_data
    ]

    print(
        "\n"
        + tabulate(formatted_data, headers=("Concept", "Feature importance", "Splits"))
    )


def change_layers(model):
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = nn.Linear(2048, 10, bias=True)
    return model


def train_decision_tree(
    model: DecisionTreeClassifier,
    dataset: TensorDataset,
    test_split: float,
    feature_names: List[str],
    epochs: int = 20,
    result_path: str = "results",
    figure_path: str = "results",
    filename: str = "decision_tree.json",
    show: bool = False,
    verbose: bool = False,
):
    results = []
    total_accuracy = 0

    y_test = None
    y_pred = None

    for _ in range(epochs):
        test_size = int(len(dataset) * test_split)
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        if verbose:
            print(f"Training model with {len(train_dataset)} samples")
            print(f"Testing model with {len(test_dataset)} samples")

        X_train = train_dataset[:][0].numpy()
        y_train = train_dataset[:][1].numpy()

        X_test = test_dataset[:][0].numpy()
        y_test = test_dataset[:][1].numpy()

        num_train_samples = X_train.shape[0]
        X_train_reshaped = X_train.reshape(num_train_samples, -1)

        num_test_samples = X_test.shape[0]
        X_test_reshaped = X_test.reshape(num_test_samples, -1)

        model.fit(X_train_reshaped, y_train)
        y_pred = model.predict(X_test_reshaped)
        accuracy = accuracy_score(y_test, y_pred)
        total_accuracy += accuracy
        if verbose:
            print(f"Test set accuracy: {accuracy:.4f}")

        feature_names += ["random"]
        plt.figure(figsize=(20, 10))
        plot_tree(
            model,
            filled=True,
            feature_names=feature_names,  # class_names=data.target_names
        )
        if show:
            plt.show()
        image_filename = "image_" + filename.replace(".json", ".png")
        plt.savefig(os.path.join(figure_path, image_filename))

        feature_importances = model.feature_importances_
        feature_split_counts = np.zeros(len(feature_names))

        def count_feature_splits(node):
            if node == -1:
                return
            feature = model.tree_.feature[node]
            if feature != -2:
                feature_split_counts[feature] += 1
                count_feature_splits(model.tree_.children_left[node])
                count_feature_splits(model.tree_.children_right[node])

        count_feature_splits(0)

        result = {}
        for name, importance, count in zip(
            feature_names, feature_importances, feature_split_counts
        ):
            result[name] = [float(importance), float(count)]
        results.append(result)

    average_results = defaultdict(list[float])
    for key in results[0].keys():
        average_results[key] = [0, 0]

    for result in results:
        for key, value in result.items():
            importance, splits = value[0], value[1]
            average_results[key][0] += importance
            average_results[key][1] += splits
    for key in average_results.keys():
        average_results[key][0] /= epochs
        average_results[key][1] /= epochs

    average_accuracy = total_accuracy / epochs

    log_decision_tree_feature_importance(average_results)

    """
    path = os.path.join(result_path, filename)
    write_results(average_results, path)
    log_decision_tree_feature_importance(average_results)

    report = classification_report(y_test, y_pred, output_dict=True)
    write_results(
        report, os.path.join(result_path, "classification_report_" + filename)
    )
    if verbose:
        print(report)
    """

    print(f"\nTree Depth: {model.get_depth()}")
    print(f"Number of Leaves: {model.get_n_leaves()}")
    print(f"Average Test set Accuracy: {average_accuracy:.4f}")

    path = os.path.join(result_path, "info_" + filename)
    result = {
        "tree_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "accuracy": average_accuracy,
    }

    """
    write_results(result, path)
    """

    return model


def compare_matrices_ssim(A, B):
    A = np.array(A, dtype=int).squeeze()
    B = np.array(B, dtype=int).squeeze()
    score = ssim(A, B)
    return score


def compare_matrices_abs(A, B):
    obs_diff = torch.abs(A - B).mean().sum()
    return obs_diff


def get_concept_activations(
    observation: Dict[str, List],
    models: Dict[str, nn.Module],
    ignore_layers: List = [],
) -> tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
]:
    activations = {}
    inputs = {}
    outputs = {}

    for key in observation.keys():
        activation, input, output = compute_activations_from_models(
            models, observation[key], ignore_layers
        )
        activations[key] = activation
        inputs[key] = input
        outputs[key] = output
    return activations, inputs, outputs


def compute_activations_from_models(
    artifacts: Dict[str, nn.Module], input: List, ignore: List[str] = []
) -> tuple[Dict[str, Dict], Dict[str, List], Dict[str, torch.Tensor]]:
    activations = {}
    inputs = {}
    outputs = {}
    for key, model in artifacts.items():
        activation_tracker = ActivationTracker(model, ignore)
        _activations, _input, _output = activation_tracker.compute_activations(input)
        activation_tracker.clean()

        activations[key] = _activations
        inputs[key] = _input
        outputs[key] = _output
    return activations, inputs, outputs


class ActivationTracker:
    def __init__(self, model: nn.Module, ignore: List[str] = []):
        self._model = copy.deepcopy(model)
        self._activations = {}
        self._hook_handles = []
        self._register_hooks(ignore)
        self._key = 0

    def compute_activations(self, inputs: List) -> tuple[dict, List, torch.Tensor]:
        self._activations.clear()
        self._key = 0
        inputs = torch.tensor(inputs).unsqueeze(1)
        outputs = self._model(inputs)
        activations_cloned = {key: value for key, value in self._activations.items()}
        return activations_cloned, inputs, outputs

    def clean(self):
        self._remove_hooks()

    def _register_hooks(self, ignore: List[str]):
        hook_count = 0
        for name, layer in self._model.named_children():
            if name in ignore:
                # logging.info(
                #     f"Ignoring layer: {processor_name} - {layer_name} - {sub_layer_name}"
                # )
                continue
            if not any(handle == name for handle in self._hook_handles):
                handle = layer.register_forward_hook(self._module_hook)
                self._hook_handles.append(handle)
                hook_count += 1
        assert hook_count > 0, "No hooks registered"

    def _remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _module_hook(self, module: nn.Module, input, output):
        self._activations[str(self._key) + "-" + str(module)] = {
            "input": input[0],
            "output": output,
        }
        self._key += 1


def get_probes_and_activations(
    ignore_layers: List[str],
    models: Dict[str, nn.Module],
    positive_observations: Dict[str, np.ndarray],
    negative_observations: Dict[str, np.ndarray],
) -> tuple[
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    probes = {}
    positive_activations = {}
    negative_activations = {}

    for concept in positive_observations.keys():
        positive_observation = positive_observations[concept]
        negative_observation = negative_observations[concept]
        probe, positive_activation, negative_activation = get_probes(
            models, positive_observation, negative_observation, ignore_layers
        )
        probes[concept] = probe
        positive_activations[concept] = positive_activation
        negative_activations[concept] = negative_activation

    return probes, positive_activations, negative_activations


def get_probes(
    models: Dict[str, nn.Module],
    positive_observation: Any,
    negative_observation: Any,
    ignore: List["str"] = [],
) -> tuple[Dict[str, Dict[str, Any]], Dict, Dict]:
    regressors = {}
    positive_activations = {}
    negative_activations = {}

    for model_name, model in models.items():
        linear_probe = LinearProbe(
            model,
            positive_observation,
            negative_observation,
            ignore,
        )
        model_regressors, positive_activation, negative_activation = (
            linear_probe.train()
        )
        regressors[model_name] = model_regressors
        positive_activations[model_name] = positive_activation
        negative_activations[model_name] = negative_activation

    return regressors, positive_activations, negative_activations


class LinearProbe:
    def __init__(
        self,
        model: nn.Module,
        positive_observations: Any,
        negative_observations: Any,
        ignore: List[str] = [],
    ):
        self._model = model
        self._model.eval()

        self._positive_observations = positive_observations
        self._negative_observations = negative_observations

        self._activation_tracker = ActivationTracker(self._model, ignore)

        np.random.seed(None)

    def train(self) -> tuple[Dict[str, LogisticRegression], Dict, Dict]:
        positive_observations = self._positive_observations
        negative_observations = self._negative_observations

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


def preprocess_activations(activations: dict) -> np.ndarray:
    numpy_activations = activations["output"].detach().numpy()
    reshaped_activations = numpy_activations.reshape(numpy_activations.shape[0], -1)
    return reshaped_activations


def get_concept_scores(
    concepts: List[str],
    test_activations: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    probes: Dict[str, Dict[str, Dict[str, LogisticRegression]]],
    results_path: str = "results",
    figure_path: str = "results",
    show: bool = False,
) -> Dict[str, Dict[str, float]]:
    concept_scores = {}

    for concept in concepts:
        concept_score = binary_concept_scores(
            test_activations[concept], probes[concept]
        )
        concept_scores[concept] = concept_score

        plot_3d(
            concept_score,
            label=concept,
            folder_path=figure_path,
            filename=f"concept_score_{concept}",
            min=0,
            max=1,
            show=show,
        )
    write_results(concept_scores, os.path.join(results_path, "concept_scores.json"))
    return concept_scores


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


def binary_concept_score(activations: np.ndarray, probe: LogisticRegression) -> float:
    labels = np.ones(activations.shape[0])
    score = 2 * max(probe.score(activations, labels) - 0.5, 0)
    return score


def plot_3d(
    X: Dict[str, Dict[str, Any]],
    filename: str = "plot3d",
    folder_path: str = "plots",
    label: str = "",
    title: str = "Plot",
    show: bool = True,
    min: float | None = None,
    max: float | None = None,
):
    X = {
        key: X[key]
        for key in sorted(
            X.keys(), key=lambda x: [int(i) for i in re.findall(r"\d+", x)]
        )
    }
    os.makedirs(folder_path, exist_ok=True)
    save_path = f"{folder_path}/{filename}.png"

    x_keys = X.keys()
    y_keys = next(iter(X.values())).keys()
    matrix = np.array([[X[x][y] for y in y_keys] for x in x_keys])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    _x = np.arange(matrix.shape[1])
    _y = np.arange(matrix.shape[0])
    _xx, _yy = np.meshgrid(_x, _y)

    min = matrix.min() if min is None else min
    max = matrix.max() if max is None else max

    cmap = cm.get_cmap("plasma")
    ax.plot_surface(
        _xx,
        _yy,
        matrix,
        edgecolor="k",
        cmap=cmap,
        alpha=1.0,
        shade=True,
    )

    ax.set_title(f"{title} - {label}", fontsize=16)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Steps", fontsize=14)
    ax.set_zlabel("Score", fontsize=14)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(y_keys, rotation=45)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(x_keys)
    ax.set_zlim(min, max)

    if show:
        plt.show()
    fig.savefig(save_path)
    plt.close(fig)


def get_tcav_scores(
    concepts: List[str],
    test_activations: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    test_output: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    probes: Dict[str, Dict[str, Dict[str, LogisticRegression]]],
    results_path: str = "results",
    figure_path: str = "results",
    show: bool = False,
) -> Dict[str, Dict[str, float]]:
    concept_tcav_scores = {}

    for concept in concepts:
        scores = tcav_scores(
            test_activations[concept], test_output[concept], probes[concept]
        )
        concept_tcav_scores[concept] = scores

        plot_3d(
            scores,
            label=concept,
            folder_path=figure_path,
            filename="tcav_" + str(concept),
            min=0,
            max=1,
            show=show,
        )
    write_results(
        concept_tcav_scores,
        os.path.join(results_path, "tcav_scores.json"),
    )
    return concept_tcav_scores


def tcav_scores(
    activations: Dict[str, Dict[str, Dict]],  # activations[model_name][layer_name]
    network_output: Dict[str, torch.Tensor],
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
            cav = probe.coef_
            scores[model_name][layer_name] = _tcav_score(
                layer_activation, network_output[model_name], cav
            )
    return scores


def _tcav_score(
    activations: Dict[str, torch.Tensor], network_output: torch.Tensor, cav: np.ndarray
) -> float:
    torch_activations = activations["output"]
    assert isinstance(torch_activations, torch.Tensor), "Activations must be a tensor"
    assert torch_activations.requires_grad, "Activations must have requires_grad=True"
    outputs = [network_output[..., i] for i in range(len(network_output[0]))]
    tcav_score = 0
    for output in outputs:
        sensitivity_score = _sensitivity_score(torch_activations, output, cav)
        mean_sensitivity_score = (sensitivity_score > 0).mean()
        tcav_score += mean_sensitivity_score
    return tcav_score / len(outputs)


def _sensitivity_score(
    activations: torch.Tensor, network_output: torch.Tensor, cav: np.ndarray
) -> np.ndarray:
    assert cav.ndim == 2, "Coef must be 2D (n_features, n_classes)"

    grads = calculate_gradients(activations, network_output)

    grads_flattened = grads.contiguous().view(grads.size(0), -1).detach().numpy()
    # grads_flattened = grads.view(grads.size(0), -1).detach().numpy() # Removed due to grads may not be contiguous
    sensitivity_score = np.dot(grads_flattened, cav.T)

    return sensitivity_score


def calculate_gradients(
    variable: torch.Tensor, target: torch.Tensor, allow_unused=False
) -> torch.Tensor:
    assert isinstance(variable, torch.Tensor), "variable must be a tensor"
    assert isinstance(target, torch.Tensor), "target must be a tensor"
    assert variable.requires_grad, "variable must have requires_grad=True"
    assert target.requires_grad, "target must have requires_grad=True"

    grads = torch.autograd.grad(
        target,
        variable,
        grad_outputs=torch.ones_like(target),
        create_graph=True,
        retain_graph=True,
        allow_unused=allow_unused,
    )[0]

    return grads


class IndexedEnum(enum.Enum):
    """
    Enum where each member has a corresponding integer index.
    """

    def __int__(self):
        return self.to_index()

    @classmethod
    def add_item(cls, name: str, value: Any):
        """
        Add a new enumeration member.
        """
        enum.extend_enum(cls, name, value)
        _enum_array.cache_clear()
        _enum_index.cache_clear()

    @classmethod
    def from_index(cls, index: int) -> str:
        """
        Get the enumeration member corresponding to a given index.
        """
        out = _enum_array(cls)[index]
        return cls(out) if out.ndim == 0 else out

    def to_index(self) -> int:
        """
        Get the index of this enumeration member.
        """
        return _enum_index(self)


def convert_numpy_to_float(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, np.generic):
            d[key] = value.item()
        elif isinstance(value, dict):
            d[key] = convert_numpy_to_float(value)
    return d


def get_combinations(data: List):
    combinations = []
    for r in range(1, len(data) + 1):
        combinations.extend(itertools.combinations(data, r))

    result_combinations = [list(comb) for comb in combinations]
    return result_combinations


def log_shapley_values(shapley_values: Dict):
    table_data = dict(
        sorted(shapley_values.items(), key=lambda item: item[1], reverse=True)
    )
    logging.info(
        "\n" + tabulate(table_data.items(), headers=("Concept", "Shapley value"))
    )
