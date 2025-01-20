import torch
import torch.nn.functional as F
from typing import Tuple


def ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratios = torch.exp(new_log_probs - old_log_probs)
    advantages = advantages.squeeze(-1)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_loss = -(new_log_probs * torch.exp(new_log_probs)).mean()
    return policy_loss, value_loss, entropy_loss


def compute_log_probs(
    actions: torch.Tensor, action_logits: torch.Tensor
) -> torch.Tensor:
    dist = torch.distributions.Categorical(logits=action_logits)
    log_probs = dist.log_prob(actions)
    return log_probs
