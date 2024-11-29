from multigrid.base import MultiAgentEnv
from multigrid.core.grid import Grid


class EmptyEnv(MultiAgentEnv):
    def _gen_grid(self):
        self.grid = Grid()
