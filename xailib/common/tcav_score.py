from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression


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
    assert isinstance(activations, torch.Tensor), "Activations must be a tensor"
    assert activations.requires_grad, "Activations must have requires_grad=True"
    assert cav.ndim == 2, "Coef must be 2D (n_features, n_classes)"

    grads = torch.autograd.grad(
        network_output,
        activations,
        grad_outputs=torch.ones_like(network_output),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads_flattened = grads.contiguous().view(grads.size(0), -1).detach().numpy()
    # grads_flattened = grads.view(grads.size(0), -1).detach().numpy() # Removed due to grads may not be contiguous
    sensitivity_score = np.dot(grads_flattened, cav.T)

    return sensitivity_score
