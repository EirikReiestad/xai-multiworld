from multiworld.base import MultiWorldEnv
from multiworld.core.grid import Grid
from multiworld.core.world_object import Box, Wall, Container
from multiworld.core.area import Area
from multiworld.utils.position import Position


class EmptyEnv(MultiWorldEnv):
    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)

        container_obj = lambda: Container()
        area_size = (3, 3)
        container_area = Area(area_size, container_obj)
        container_area.place(
            self.grid,
            (
                (self.grid.width - int(area_size[0])) // 2,
                (self.grid.height - int(area_size[1])) // 2,
            ),
        )

        placeable_positions = self.grid.get_empty_positions(10)
        for pos in placeable_positions:
            self.grid.set(pos, Box())

        placeable_positions = self.grid.get_empty_positions(len(self.agents))
        for agent, pos in zip(self.agents, placeable_positions):
            agent.state.pos = pos
