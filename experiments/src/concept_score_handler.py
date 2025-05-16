import numpy as np
import torch
from experiments.src.observation_handler import obs_diff


def calc_concept_scores(average_positive_observations, all_X):
    scores = []
    for obs in all_X:
        score = concept_score(average_positive_observations, obs)
        scores.append(score)
    return np.array(scores)


def concept_score(
    cav_obs: dict[str, torch.Tensor], other_obs: torch.Tensor
) -> list[float]:
    score = []
    for key, obs in cav_obs.items():
        diff = obs_diff(obs, other_obs)
        score.append(diff)
    return score
