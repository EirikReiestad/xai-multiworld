from multigrid.base import MultiGridEnv
from multigrid.core.grid import Grid
from multigrid.core.world_object import Box, Wall
from multigrid.utils.position import Position


class EmptyEnv(MultiGridEnv):
    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)

        for agent in self.agents:
            while True:
                agent.state.pos = self._rand_pos(0, width, 0, height)
                start_cell = self.grid.get(agent.state.pos)
                if start_cell is None or start_cell.can_overlap():
                    break

        self.grid.set(self._rand_pos(0, width, 0, height), Box())
        for _ in range(10):
            self.grid.set(self._rand_pos(0, width, 0, height), Wall())
