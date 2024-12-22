import torch.nn as nn
import numpy as np
import copy
import torch
from typing import List


class ActivationTracker:
    def __init__(self, model: nn.Module):
        self._model = model
        self._activations = {}

    def compute_activations(self, inputs: List) -> tuple[dict, torch.Tensor]:
        self._activations.clear()
        outputs = self._model(*inputs)
        activations = copy.deepcopy(self._activations)
        return activations, outputs

    def _register_hooks(self):
        for _, processor in self._model.named_children():
            for _, layer in processor.named_children():
                for name, sub_layer in layer.named_children():
                    if not isinstance(sub_layer, nn.ReLU):
                        continue
                    sub_layer.register_forward_hook(self._module_hook)

    def _module_hook(self, module: nn.Module, input, output):
        self._activations[module] = {
            "input": input[0],
            "output": output,
        }


def preprocess_activations(activations: dict) -> np.ndarray:
    numpy_activations = activations["output"].detach().numpy()
    reshaped_activations = numpy_activations.reshape(numpy_activations.shape[0], -1)
    return reshaped_activations
