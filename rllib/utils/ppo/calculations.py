import torch
import numpy as np
from numpy.typing import NDArray
from typing import List


def ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss
    # NOTE: There are other improvements to the loss that I believe codebases like stable-baselines3 implement, which include value loss and entropy loss.
    # Might implement later. Suggestion:
    # value_loss = F.mse_loss(values, returns)
    # entropy_loss = -entropy.mean()
    # return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss


def compute_log_probs(
    actions: torch.Tensor, action_probs: torch.Tensor
) -> torch.Tensor:
    selected_action_probs = action_probs.gather(dim=-1, index=actions.unsqueeze(-1))
    log_probs = torch.log(selected_action_probs)
    return log_probs
