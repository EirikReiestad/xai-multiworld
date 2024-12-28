from typing import List

import torch


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
