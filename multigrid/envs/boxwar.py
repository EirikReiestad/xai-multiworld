import random
from typing import Any, List, SupportsFloat

import numpy as np
from numpy.typing import NDArray

from multigrid.base import MultiGridEnv
from multigrid.core.action import Action
from multigrid.core.agent import Agent
from multigrid.core.area import Area
from multigrid.core.constants import Color
from multigrid.core.grid import Grid
from multigrid.core.world_object import Box, Container, Goal, Wall, WorldObject
from multigrid.utils.position import Position
from multigrid.utils.typing import AgentID, ObsType


class BoxWarEnv(MultiGridEnv):
    def __init__(self, boxes: int, *args, **kwargs):
        self._num_boxes = boxes
        self._num_teams = 2
        super().__init__(*args, **kwargs)

        self._team_score = {}

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)

        container_obj = lambda: Container(color=Color.blue)
        area_size = (self._width // 2 - 1, self._height)
        container_area = Area(area_size, container_obj)
        container_area.place(self.grid, (0, 0))

        container_obj = lambda: Container(color=Color.red)
        container_area = Area(area_size, container_obj)
        container_area.place(self.grid, (self._width - area_size[0], 0))

        placeable_positions = self.grid.get_empty_positions(self._num_boxes)
        for pos in placeable_positions:
            self.grid.set(pos, Box())

        placeable_positions = self.grid.get_empty_positions(len(self.agents))

        team_split = len(self.agents) // self._num_teams
        team_colors = ["blue", "red"]

        for i, (agent, pos) in enumerate(zip(self.agents, placeable_positions)):
            agent.state.pos = pos
            agent.color = team_colors[i // team_split]

    def reset(self):
        self._team_score = {}
        return super().reset()

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
            if actions[agent.index] == Action.drop:
                self._handle_drop(agent, rewards, terminations)
            if actions[agent.index] == Action.pickup:
                self._handle_pickup(agent, rewards, terminations)

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

    def _handle_pickup(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
    ):
        if agent.state.carrying is not None:
            return

        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(fwd_pos)

        if fwd_obj is None:
            return

        if not isinstance(fwd_obj, Container):
            return

        if fwd_obj.contains is None:
            return

        agent_present = np.array(self._agent_states.pos == fwd_pos).any()
        if agent_present:
            return

        self._team_score[fwd_obj.color.to_index()] = (
            self._team_score.get(fwd_obj.color.to_index(), 0) - 1
        )

        team_captain = self._get_team_captain(fwd_obj.color)
        self.add_reward(team_captain, rewards, -1, joint_reward=False, team_reward=True)

    def _handle_drop(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
    ):
        if agent.state.carrying is None:
            return

        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(fwd_pos)

        if fwd_obj is None:
            return

        if not isinstance(fwd_obj, Container):
            return

        if fwd_obj.contains is not None:
            return

        agent_present = np.array(self._agent_states.pos == fwd_pos).any()
        if agent_present:
            return

        self._team_score[fwd_obj.color.to_index()] = (
            self._team_score.get(fwd_obj.color.to_index(), 0) + 1
        )
        self._reward_team(fwd_obj.color, rewards, 1)

        if self._team_score[fwd_obj.color.to_index()] == self._num_boxes:
            team_captain = self._get_team_captain(fwd_obj.color)
            self.on_success(
                team_captain,
                rewards,
                terminations,
            )

    def _get_team_captain(self, color: Color) -> Agent:
        for agent in self.agents:
            if agent.color == color:
                return agent
        raise ValueError("No team captain found")

    def _reward_team(
        self, color: Color, rewards: dict[AgentID, SupportsFloat], reward: float
    ):
        for agent in self.agents:
            if agent.color == color:
                self.add_reward(
                    agent, rewards, reward, joint_reward=False, team_reward=True
                )
                break
