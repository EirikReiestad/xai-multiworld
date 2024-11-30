from multigrid.base import MultiAgentEnv
from multigrid.core.grid import Grid


class EmptyEnv(MultiAgentEnv):
    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
