import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def maximize_activation(
    probe: LogisticRegression,
    activation: torch.Tensor,
    lambda_: float = 1,
) -> torch.Tensor:
    reshaped_activation = activation.reshape(1, -1)

    intercept = torch.tensor(probe.intercept_, dtype=torch.float32, requires_grad=True)

    norm = np.linalg.norm(probe.coef_)
    norm_coef = torch.tensor(probe.coef_ / norm, dtype=torch.float32)

    max_activation = torch.mul(reshaped_activation, norm_coef * lambda_) + intercept
    max_activation = max_activation.view(activation.shape)
    return max_activation
