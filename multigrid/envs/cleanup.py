from typing import Any, SupportsFloat

import numpy as np

from multigrid.base import MultiGridEnv
from multigrid.core.action import Action
from multigrid.core.grid import Grid
from multigrid.core.world_object import Box, Goal, Wall, WorldObject, Container
from multigrid.utils.position import Position
from multigrid.utils.typing import AgentID, ObsType
from multigrid.core.area import Area


class CleanUpEnv(MultiGridEnv):
    def __init__(self, boxes: int, *args, **kwargs):
        self._num_boxes = boxes
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)

        container_obj = lambda: Container()
        container_area = Area((2, 2), container_obj)

        container_area.place(self.grid, (0, 0))

        placeable_positions = self.grid.get_empty_positions(self._num_boxes)
        for pos in placeable_positions:
            self.grid.set(pos, Box())

        placeable_positions = self.grid.get_empty_positions(len(self.agents))
        for agent, pos in zip(self.agents, placeable_positions):
            agent.state.pos = pos

    def step(
        self, actions: dict[AgentID, Action | int]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, SupportsFloat],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        observations, rewards, terminations, truncations, info = super().step(actions)
        for agent in self.agents:
            obj = self.grid.get(agent.state.pos)
            if obj is None:
                continue
            if np.array_equal(obj, self.goal):
                agent.pos = Position(-1, -1)
        return observations, rewards, terminations, truncations, info
