from multigrid.utils.typing import AgentID, ObsType


def preprocess_next_observations(
    next_observations: dict[AgentID, ObsType],
    terminations: dict[AgentID, bool],
    truncations: dict[AgentID, bool],
) -> dict[AgentID, ObsType]:
    next_obs = {}
    for agent_id, next_observation in next_observations.items():
        if terminations[agent_id] or truncations[agent_id]:
            next_obs[agent_id] = None
        else:
            next_obs[agent_id] = next_observation
    return next_obs
