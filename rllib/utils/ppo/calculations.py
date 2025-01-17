import torch
import torch.nn.functional as F


def ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    epsilon: float,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
) -> torch.Tensor:
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values.view(-1), returns.float().view(-1))
    entropy_loss = -(new_log_probs * torch.exp(new_log_probs)).mean()
    return policy_loss + value_weight * value_loss # + entropy_weight * entropy_loss


def compute_log_probs(
    actions: torch.Tensor, action_probs: torch.Tensor
) -> torch.Tensor:
    selected_action_probs = action_probs.gather(dim=-1, index=actions.unsqueeze(-1))
    log_probs = torch.log(selected_action_probs)
    return log_probs
