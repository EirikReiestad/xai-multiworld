import torch.nn as nn
import numpy as np
import copy
import torch
from typing import List


class ActivationTracker:
    def __init__(self, model: nn.Module):
        self._model = model
        self._activations = {}

        self._register_hooks()

    def compute_activations(self, inputs: List) -> tuple[dict, List, torch.Tensor]:
        self._activations.clear()
        outputs = self._model(*inputs)
        activations_cloned = {key: value for key, value in self._activations.items()}
        return activations_cloned, inputs, outputs

    def _register_hooks(self):
        hook_count = 0
        for _, processor in self._model.named_children():
            for _, layer in processor.named_children():
                for name, sub_layer in layer.named_children():
                    if not isinstance(sub_layer, nn.ReLU):
                        continue
                    hook_count += 1
                    sub_layer.register_forward_hook(self._module_hook)

        assert hook_count > 0, "No hooks registered"

    def _module_hook(self, module: nn.Module, input, output):
        self._activations[module] = {
            "input": input[0],
            "output": output,
        }


def preprocess_activations(activations: dict) -> np.ndarray:
    numpy_activations = activations["output"].detach().numpy()
    reshaped_activations = numpy_activations.reshape(numpy_activations.shape[0], -1)
    return reshaped_activations
