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

    value_loss = F.mse_loss(returns, values)
    entropy_loss = -(new_log_probs * torch.exp(new_log_probs)).mean()
    return policy_loss, value_loss, entropy_loss


def compute_log_probs(
    actions: torch.Tensor,
    action_logits: torch.Tensor,
    std: float = 1.0,
    continuous: bool = True,
):
    if continuous:
        dist = torch.distributions.Normal(action_logits, std)
        log_probs = dist.log_prob(actions.unsqueeze(-1)).sum(-1)
    else:
        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions.unsqueeze(-1))
    return log_probs


def compute_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards).float()
    returns[:, -1] = rewards[:, -1]
    for t in reversed(range(rewards.size(1) - 1)):
        returns[:, t] = rewards[:, t] + gamma * returns[:, t + 1]
    return returns
