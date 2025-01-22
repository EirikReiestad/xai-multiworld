from dataclasses import dataclass
from typing import Dict, SupportsFloat, Any

from multigrid.utils.serialization import deserialize_observation, serialize_observation
from multigrid.utils.typing import AgentID, ObsType


@dataclass
class Observations:
    observations: Dict[AgentID, ObsType]
    actions: Dict[AgentID, int]
    rewards: Dict[AgentID, SupportsFloat]
    terminations: Dict[AgentID, bool]
    truncations: Dict[AgentID, bool]
    infos: Dict[AgentID, Dict[str, Any]]

    def serialize(self):
        return {
            "observations": {
                agent_id: serialize_observation(obs)
                for agent_id, obs in self.observations.items()
            },
            "actions": self.actions,
            "rewards": self.rewards,
            "terminations": self.terminations,
            "truncations": self.truncations,
            "infos": {
                agent_id: {
                    key: str(value)
                    if isinstance(value, (int, float, str))
                    else str(value)
                    for key, value in info.items()
                }
                for agent_id, info in self.infos.items()
            },
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> "Observations":
        return Observations(
            observations={
                agent_id: deserialize_observation(obs)
                for agent_id, obs in data["observations"].items()
            },
            actions={
                agent_id: int(action) for agent_id, action in data["actions"].items()
            },
            rewards={
                agent_id: float(reward) for agent_id, reward in data["rewards"].items()
            },
            terminations={
                agent_id: bool(termination)
                for agent_id, termination in data["terminations"].items()
            },
            truncations={
                agent_id: bool(truncation)
                for agent_id, truncation in data["truncations"].items()
            },
            infos={
                agent_id: {
                    key: value if isinstance(value, (int, float, str)) else value
                    for key, value in info.items()
                }
                for agent_id, info in data["infos"].items()
            },
        )
