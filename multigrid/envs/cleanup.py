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
        for agent in self.agents:
            if actions[str(agent.index)] != Action.drop:
                continue

            if agent.state.carrying is None:
                continue

            fwd_pos = agent.front_pos
            fwd_obj = self.grid.get(fwd_pos)
            if fwd_obj is not None and not fwd_obj.can_place:
                continue

            agent_present = np.array(self._agent_states.pos == fwd_pos).any()
            if agent_present:
                continue

        observations, rewards, terminations, truncations, info = super().step(actions)

        return observations, rewards, terminations, truncations, info
