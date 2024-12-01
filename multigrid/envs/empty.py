from multigrid.base import MultiAgentEnv
from multigrid.core.grid import Grid


class EmptyEnv(MultiAgentEnv):
    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        for agent in self.agents:
            agent.state.pos = self._rand_pos(0, width, 0, height)
