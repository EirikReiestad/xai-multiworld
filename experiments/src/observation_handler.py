import torch
from experiments.src.compute_statistics import pearson_correlation


def obs_diff(obs: torch.Tensor, other_obs: list[torch.Tensor]) -> float:
    diffs = [pearson_correlation(obs, o) for o in other_obs]
    # diffs = [compare_matrices_abs(obs, o) for o in other_obs]
    obs_diff = sum(diffs)
    max_diff = len(other_obs)
    diff = obs_diff / max_diff
    return float(diff)
