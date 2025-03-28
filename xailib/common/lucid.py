import os
from typing import Dict

import torch
from lucent.model_utils import get_model_layers
from lucent.optvis import objectives, render


def lucid_visualization(
    model: torch.nn.Module,
    cavs: Dict[str, torch.Tensor],
    layer_name: str | None = None,
    result_path: str = os.path.join("assets", "results"),
):
    layer_names, dependency_graph = get_model_layers(model)

    layer_name = layer_name or layer_names[-1]

    for i, cav in cavs.items():
        torch_cav = torch.tensor(cav)
        obj = objectives.direction(layer_name, cav)
        image_name = os.path.join(result_path, f"{layer_name}_{i}.png")
        list_of_images = render.render_vis(model, obj)
