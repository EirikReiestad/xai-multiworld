from typing import Any, SupportsFloat

import numpy as np

from multigrid.base import MultiGridEnv
from multigrid.core.action import Action
from multigrid.core.grid import Grid
from multigrid.core.world_object import Box, Goal, Wall, WorldObject
from multigrid.utils.position import Position
from multigrid.utils.typing import AgentID, ObsType


class GoToGoalEnv(MultiGridEnv):
    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)

        goal_pos = Position(width // 2, height // 2)
        self.goal = Goal()
        self.grid.set(goal_pos, self.goal)

        for agent in self.agents:
            placeable_positions = self.grid.get_placeable_positions()
            pos = self._rand_elem(placeable_positions)
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
