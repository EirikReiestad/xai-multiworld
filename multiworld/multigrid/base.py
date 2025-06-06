import math
from functools import partial
from typing import Dict, List, Literal, Optional, SupportsFloat, Tuple

import numpy as np
from numpy.typing import NDArray

from multiworld.base import MultiWorldEnv
from multiworld.core.position import Position
from multiworld.multigrid.core.action import Action
from multiworld.multigrid.core.agent import Agent, AgentState
from multiworld.multigrid.core.constants import TILE_PIXELS, Direction, WorldObjectType
from multiworld.multigrid.core.grid import Grid
from multiworld.multigrid.core.world_object import Container, WorldObject
from multiworld.multigrid.utils.decoder import decode_observation
from multiworld.multigrid.utils.observation import gen_obs_grid_encoding
from multiworld.multigrid.utils.ohe import decode_direction, ohe_direction
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from multiworld.multigrid.utils.render import add_highlighted_border
from multiworld.utils.typing import AgentID, ObsType
from utils.common.callbacks import RenderingCallback, empty_rendering_callback
from utils.core.constants import Color


class MultiGridEnv(MultiWorldEnv):
    def __init__(
        self,
        agents: int = 1,
        width: int = 20,
        height: int = 20,
        max_steps: int = 100,
        agent_view_size: int | None = 7,
        highlight: bool = False,
        see_through_walls: bool = False,
        joint_reward: bool = False,
        team_reward: bool = False,
        preprocessing: PreprocessingEnum = PreprocessingEnum.none,
        tile_size=TILE_PIXELS,
        screen_size: Tuple[int, int] | None = None,
        render_mode: Literal["human", "rgb_array"] = "human",
        rendering_callback: RenderingCallback = empty_rendering_callback,
        success_termination_mode: Literal["all", "any"] = "all",
        failure_termination_mode: Literal["all", "any"] = "any",
        caption: str = "MultiGrid",
    ):
        if screen_size is None:
            screen_size = (width * tile_size, height * tile_size)
        elif isinstance(screen_size, int):
            screen_size = (screen_size, screen_size)
            tile_size = min(screen_size) // max(width, height)

        super().__init__(
            agents,
            width,
            height,
            max_steps,
            joint_reward,
            team_reward,
            screen_size,
            render_mode,
            rendering_callback,
            caption,
            success_termination_mode,
            failure_termination_mode,
        )
        self.metadata["name"] = "multigrid"
        self._preprocessing = preprocessing
        self._highlight = highlight
        self._tile_size = tile_size
        self._render_size = None

        self._screen_size = screen_size
        assert isinstance(screen_size, tuple)

        self._agent_view_size = agent_view_size
        self._agent_see_through_walls = see_through_walls
        self._agent_states = AgentState(agents)
        self._agents: List[Agent] = []
        for i in range(self._num_agents):
            agent = Agent(
                i, agent_view_size or self._width, see_through_walls, preprocessing
            )
            self._agents.append(agent)
        self._world = Grid(width, height)

        self._render_direction = None
        self._border_size = 20

    def update_from_numpy(self, observation: NDArray):
        grid: NDArray[np.int_] = observation[0]
        decode_obs = partial(decode_observation, preprocessing=self._preprocessing)
        obs = decode_obs({"observation": grid})
        grid = obs["observation"]
        direction = decode_direction(observation[1])
        self._render_direction = Direction(direction)
        assert grid.ndim == 3, "Input grid must be 3-dimensional."
        height, width, dim = grid.shape
        assert (
            dim == WorldObject.dim
        ), f"Last dimension must match WorldObject.dim ({WorldObject.dim})."

        self._agents = []

        for j in range(height):  # Iterate over rows
            for i in range(width):  # Iterate over columns
                pos = Position(i, j)
                grid_obj = grid[pos.y, pos.x]
                type_idx = grid_obj[WorldObject.TYPE]
                if type_idx == WorldObjectType.agent.to_index():
                    previous_agent_idx = (
                        self._agents[-1].index if len(self._agents) > 0 else 0
                    )
                    agent = Agent(
                        previous_agent_idx, self._agent_view_size or self._width
                    )
                    agent.state.dir = self._rand_int(0, 4)
                    agent.pos = pos
                    self._agents.append(agent)
                    grid[pos.y, pos.x] = WorldObjectType.empty.to_index()
                    continue

        self._world = Grid.from_numpy(grid)

    def place_agent(
        self, agent: Agent, top=None, size=None, rand_dir=True, max_tries=math.inf
    ) -> Position:
        """
        Set agent starting point at an empty position in the grid.
        """
        agent.state.pos = Position(-1, -1)
        pos = self._place_object(None, top, size, max_tries=max_tries)
        agent.state.pos = pos

        if rand_dir:
            agent.state.dir = self._rand_int(0, 4)

        return pos

    @property
    def agents(self) -> List[Agent]:
        return self._agents

    @property
    def agent_states(self) -> AgentState:
        return self._agent_states

    @property
    def world(self) -> Grid:
        return self._world

    def _execute_action(
        self, agent: Agent, action: Action | int, rewards: Dict[AgentID, SupportsFloat]
    ) -> None:
        if agent.state.terminated:
            return

        # Rotate left
        if action == Action.left:
            agent.state.dir = (agent.dir - 1) % 4

        # Rotate right
        elif action == Action.right:
            agent.state.dir = (agent.dir + 1) % 4

        # Move forward
        elif action == Action.forward:
            fwd_pos = agent.front_pos
            if not self._world.in_bounds(fwd_pos):
                return

            fwd_obj = self._world.get(fwd_pos)

            if fwd_obj is not None and not fwd_obj.can_overlap():
                return

            agent_present = np.array(self._agent_states.pos == fwd_pos).any()
            if agent_present:
                return

            agent.state.pos = fwd_pos
            if fwd_obj is not None:
                if fwd_obj.type == WorldObjectType.goal:
                    self.on_success(agent, rewards, {})

        elif action == Action.pickup:
            if agent.state.carrying is not None:
                return

            fwd_pos = agent.front_pos
            fwd_obj = self._world.get(fwd_pos)

            if fwd_obj is None:
                return

            if isinstance(fwd_obj, Container):
                if fwd_obj.can_pickup_contained() is False:
                    return
                agent.state.carrying = fwd_obj.contains
                fwd_obj.contains = None
                return

            if not fwd_obj.can_pickup():
                return

            agent.state.carrying = fwd_obj
            self._world.set(fwd_pos, None)

        elif action == Action.drop:
            if agent.state.carrying is None:
                return

            fwd_pos = agent.front_pos
            fwd_obj = self._world.get(fwd_pos)

            if not self._world.in_bounds(fwd_pos):
                return

            agent_present = np.array(self._agent_states.pos == fwd_pos).any()
            if agent_present:
                return

            if fwd_obj is not None and fwd_obj.can_contain():
                fwd_obj.contains = agent.state.carrying
                agent.state.carrying = None
                return

            if fwd_obj is not None:
                return

            self._world.set(fwd_pos, agent.carrying)
            agent.state.carrying.cur_pos = fwd_pos
            agent.state.carrying = None

        elif action == Action.toggle:
            fwd_pos = agent.front_pos
            fwd_obj = self._world.get(fwd_pos)
            if fwd_obj is not None:
                fwd_obj.toggle(self, agent, fwd_pos)

        elif action == Action.done:
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

    def _get_frame(self) -> np.ndarray:
        return self._get_full_render(self._highlight, self._tile_size)

    def _gen_obs(self) -> Dict[AgentID, ObsType]:
        directions = self._agent_states.dir
        image = gen_obs_grid_encoding(
            self._world.state,
            self._agent_states,
            self._agent_view_size,
            self._agent_see_through_walls,
            self._preprocessing,
        )
        observations = {}
        for i in range(self._num_agents):
            ohe_dir = ohe_direction(directions[i])
            observations[i] = {
                "observation": image[i],
                "direction": ohe_dir,
            }

        return observations

    def _get_full_render(self, highlight: bool, tile_size: int) -> np.ndarray:
        if len(self._agents) > 0:
            obs_shape = self._agents[0].observation_space["observation"].shape[:-1]
        else:
            obs_shape = (self._agent_view_size, self._agent_view_size)
        vis_mask = np.zeros((self._num_agents, *obs_shape), dtype=bool)
        for key, obs in self._gen_obs().items():
            vis_mask[key] = (
                obs["observation"][..., 0] != WorldObjectType.unseen.to_index()
            )

        highlight_mask = np.zeros((self._width, self._height), dtype=bool)

        for agent in self._agents:
            if agent.state.terminated:
                continue
            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = agent.state.dir.to_vec()
            r_vec = np.array((f_vec[1], -f_vec[0]))
            top_left = (
                agent.state.pos()
                + f_vec * (agent.view_size - 1)
                - r_vec * (agent.view_size // 2)
            )

            for vis_j in range(agent.view_size):
                for vis_i in range(agent.view_size):
                    if not vis_mask[agent.index][vis_j, vis_i]:
                        pass
                        # continue
                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_i) + (r_vec * vis_j)
                    if (
                        0 <= abs_i < self._width
                        and 0 <= abs_j < self._height
                        and highlight
                    ):
                        highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self._world.render(
            tile_size, agents=self._agents, highlight_mask=highlight_mask
        )
        if self._render_direction is None:
            return img

        img = add_highlighted_border(img, self._render_direction, self._border_size)
        return img

    def _reset_agents(self):
        self._agent_states = AgentState(self._num_agents)
        for agent in self._agents:
            agent.reset()
            agent.state = self._agent_states[agent.index]
            agent.state.pos = self.place_agent(agent)

    def _get_terminations(self) -> Dict[AgentID, bool]:
        return {
            agent_id: self._agent_states[agent_id].terminated
            for agent_id in range(self._num_agents)
        }
