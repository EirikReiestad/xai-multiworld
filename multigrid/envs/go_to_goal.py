from multigrid.base import MultiGridEnv
from multigrid.core.grid import Grid
from multigrid.core.world_object import Box, Wall, Goal
from multigrid.utils.position import Position


class GoToGoalEnv(MultiGridEnv):
    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)

        goal_pos = Position(width // 2, height // 2)
        goal = Goal()
        self.grid.set(goal_pos, goal)

        for agent in self.agents:
            placeable_positions = self.grid.get_placeable_positions()
            pos = self._rand_elem(placeable_positions)
            agent.state.pos = pos
