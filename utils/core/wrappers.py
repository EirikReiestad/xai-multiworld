from typing import List
import gymnasium as gym

from multigrid.base import MultiGridEnv
from multigrid.utils.wrappers import Observations
from swarm.base import SwarmEnv

from dataclasses import dataclass
from typing import Dict, SupportsFloat, Any

from multigrid.utils.serialization import deserialize_observation, serialize_observation
from multigrid.utils.typing import AgentID, ObsType


class ObservationCollectorWrapper(gym.Wrapper):
    def __init__(
        self,
        env: MultiGridEnv | SwarmEnv,
        observations: int = 1000,
        directory: str = os.path.join("assets", "observations"),
        filename: str = "observations",
    ) -> None:
        super().__init__(env)
        self.env = env
        self._filepath = os.path.join(directory, filename + ".json")
        self._observations = observations

        self._rollouts: List[Observations] = []

    def step(
        self,
        actions: Dict[AgentID, int],
    ):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        self._rollouts.append(
            Observations(
                observations, actions, rewards, terminations, truncations, infos
            )
        )

        if len(self._rollouts) % (self._observations // 10) == 0:
            logging.info(
                f"Collcted {len(self._rollouts)} / {self._observations} observations"
            )

        if len(self._rollouts) == self._observations:
            logging.info(f"Saving rollouts to file {self._filepath}...")
            self._save_rollouts()
            sys.exit()

        return observations, rewards, terminations, truncations, infos

    def _save_rollouts(self):
        data = [rollout.serialize() for rollout in self._rollouts]

        with open(self._filepath, "w") as f:
            json.dump(data, f, indent=4)


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
