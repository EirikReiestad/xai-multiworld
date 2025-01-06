import torch
import numpy as np
from numpy.typing import NDArray


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
    return torch.tensor(float(policy_loss), requires_grad=True)
    # NOTE: There are other improvements to the loss that I believe codebases like stable-baselines3 implement, which include value loss and entropy loss.
    # Might implement later. Suggestion:
    # value_loss = F.mse_loss(values, returns)
    # entropy_loss = -entropy.mean()
    # return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss


def compute_log_probs(actions: NDArray, action_probs: NDArray) -> NDArray[np.float32]:
    assert (
        actions.shape[:2] == action_probs.shape[:2]
    ), f"actions: {actions.shape}, action_probs: {action_probs.shape}"
    log_probs = []
    for i in range(actions.shape[1]):
        log_probs.append(_compute_log_prob(actions[:, i], action_probs[:, i]))
    return np.stack(log_probs, axis=1)


def _compute_log_prob(actions: NDArray, action_probs: NDArray) -> NDArray[np.float32]:
    log_probs = []
    for action, action_prob in zip(actions, action_probs):
        prob = action_prob[int(action)]
        log_probs.append(torch.log(torch.tensor(prob, dtype=torch.float32)))
    return np.array(log_probs, dtype=np.float32)
