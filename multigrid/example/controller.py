from multigrid.utils.typing import AgentID
import numpy as np


class Controller:
    def __init__(self, agents: int = 1):
        self._agents = agents

    def get_actions(self) -> dict[AgentID, int]:
        return {str(i): np.random.choice(4) for i in range(self._agents)}
