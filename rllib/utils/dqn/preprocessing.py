import numpy as np

from multigrid.utils.typing import AgentID, ObsType


def preprocess_next_observations(
    next_observations: dict[AgentID, ObsType],
    terminations: dict[AgentID, bool],
    truncations: dict[AgentID, bool],
) -> dict[AgentID, ObsType]:
    def handle_numpy():
        return next_observations

    def handle_dict():
        next_obs = {}
        for agent_id, next_observation in next_observations.items():
            if terminations.get(agent_id) or truncations.get(agent_id):
                next_obs[agent_id] = None
            else:
                next_obs[agent_id] = next_observation
        return next_obs

    if isinstance(next_observations, np.ndarray):
        return handle_numpy()
    return handle_dict()
