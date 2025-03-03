import copy
import gc
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


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
        outputs = self._model(*inputs)
        activations_cloned = {key: value for key, value in self._activations.items()}
        return activations_cloned, inputs, outputs

    def clean(self):
        self._remove_hooks()

    def _register_hooks(self, ignore: List[str]):
        hook_count = 0
        for processor_name, processor in self._model.named_children():
            for layer_name, layer in processor.named_children():
                for sub_layer_name, sub_layer in layer.named_children():
                    if (
                        processor_name in ignore
                        or layer_name in ignore
                        or sub_layer_name in ignore
                    ):
                        # logging.info(
                        #     f"Ignoring layer: {processor_name} - {layer_name} - {sub_layer_name}"
                        # )
                        continue
                    if not isinstance(sub_layer, nn.ReLU):
                        continue
                    if not any(handle == sub_layer for handle in self._hook_handles):
                        handle = sub_layer.register_forward_hook(self._module_hook)
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


def preprocess_activations(activations: dict) -> np.ndarray:
    numpy_activations = activations["output"].detach().numpy()
    reshaped_activations = numpy_activations.reshape(numpy_activations.shape[0], -1)
    return reshaped_activations


def compute_activations_from_models(
    artifacts: Dict[str, nn.Module], input: List, ignore: List[str] = []
) -> Tuple[Dict[str, Dict], Dict[str, List], Dict[str, torch.Tensor]]:
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
