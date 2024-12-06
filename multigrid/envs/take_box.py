from multigrid.base import MultiGridEnv
from multigrid.core.grid import Grid
from multigrid.core.world_object import Box, Wall


class TakeBox(MultiGridEnv):
    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)

        for _ in range(self._num_agents * 5):
            self.grid.set(self._rand_pos(0, width, 0, height), Box())

        for agent in self.agents:
            while True:
                agent.state.pos = self._rand_pos(0, width, 0, height)
                start_cell = self.grid.get(agent.state.pos)
                if start_cell is None or start_cell.can_overlap():
                    break

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        for agent in self.agents:
            if agent.state.carrying is True:
                self.on_success(agent, reward, terminated)
        return obs, reward, terminated, truncated, info
