import random
from typing import Any, SupportsFloat

import numpy as np

from multigrid.base import MultiGridEnv
from multigrid.core.action import Action
from multigrid.core.area import Area
from multigrid.core.grid import Grid
from multigrid.core.world_object import Box, Container
from multigrid.utils.position import Position
from multigrid.utils.typing import AgentID, ObsType


class CleanUpEnv(MultiGridEnv):
    def __init__(self, boxes: int, *args, **kwargs):
        self._num_boxes = boxes
        super().__init__(*args, **kwargs)

        self._success_move_box = 0
        self._area = 0

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)

        container_obj = lambda: Container()
        area_sizes = [(1, 1), (2, 2)]
        num_areas = random.randint(1, 5)
        for _ in range(num_areas):
            area_size = self._rand_elem(area_sizes)
            placeable_areas = self.grid.get_empty_areas(area_size)
            if len(placeable_areas) == 0:
                continue
            pos: Position = self._rand_elem(placeable_areas)
            container_area = Area(area_size, container_obj)
            container_area.place(self.grid, pos())
            self._area += area_size[0] * area_size[1]

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
        rewards: dict[AgentID, SupportsFloat] = {
            agent.index: 0 for agent in self.agents
        }
        terminations: dict[AgentID, bool] = {
            agent.index: False for agent in self.agents
        }
        for agent in self.agents:
            if actions[agent.index] != Action.drop:
                continue

            if agent.state.carrying is None:
                continue

            fwd_pos = agent.front_pos
            fwd_obj = self.grid.get(fwd_pos)

            if fwd_obj is None:
                continue

            if not isinstance(fwd_obj, Container):
                continue

            if fwd_obj.contains is not None:
                continue

            agent_present = np.array(self._agent_states.pos == fwd_pos).any()
            if agent_present:
                continue

            self._success_move_box += 1
            self._area -= 1
            self.add_reward(agent, rewards, 0.1 * self._reward(), joint_reward=False)

            if self._success_move_box == self._num_boxes or self._area == 0:
                self.on_success(
                    agent,
                    rewards,
                    terminations,
                )

        observations, step_rewards, terms, truncations, info = super().step(actions)

        rewards = {
            agent.index: float(rewards[agent.index]) + float(step_rewards[agent.index])
            for agent in self.agents
        }
        terminations = {
            agent.index: bool(terminations[agent.index]) or bool(terms[agent.index])
            for agent in self.agents
        }

        return observations, rewards, terminations, truncations, info
