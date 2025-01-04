import torch
from torch.nn import functional as F
from rllib.core.memory.trajectory_buffer import Trajectory
from typing import List, Any, Dict
from numpy.typing import NDArray
from multigrid.utils.typing import AgentID


def compute_advantages(
    trajectories: List[Trajectory], gamma: float, lam: float
) -> List[float]:
    rewards, values = [], []
    for trajectory in trajectories:
        rewards.append(trajectory.reward)
        values.append(trajectory.value)
    returns, advantages = [], []
    G, A = 0, 0
    for trajectory in reversed(range(len(rewards))):
        G = rewards[trajectory] + gamma * G
        A = (
            rewards[trajectory]
            + gamma * (1 - lam) * values[trajectory + 1]
            - values[trajectory]
        )
        returns.insert(0, G)
        advantages.insert(0, A)
    returns, (advantages - advantages.mean()) / (advantages.std() + 1e-10)


def ppo_loss(
    old_log_probs: List[float],
    new_log_probs: List[float],
    advantages: List[float],
    entropy: float,
    values: Any,  # TODO: Any???
    returns: Any,  # TODO: Any?????
    epsilon: float,
) -> float:
    ratios = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_loss = -entropy.mean()
    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss


def compute_log_probs(
    actions: Dict[AgentID, int], action_probs: Dict[AgentID, NDArray]
) -> List[float]:
    log_probs = []
    for agent_id, action in actions.items():
        log_probs.append(torch.log(action_probs[agent_id][action]))
    return log_probs
