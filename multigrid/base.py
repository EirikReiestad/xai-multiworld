import math
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import repeat
from typing import Any, Callable, Literal, SupportsFloat

import gymnasium as gym
import numpy as np
import pygame as pg
from gymnasium import spaces

from multigrid.core.action import Action
from multigrid.core.agent import Agent, AgentState
from multigrid.core.constants import TILE_PIXELS, WorldObjectType
from multigrid.core.grid import Grid
from multigrid.core.world_object import WorldObject
from multigrid.utils.observation import gen_obs_grid_encoding
from multigrid.utils.position import Position
from multigrid.utils.random import RandomMixin

AgentID = str
ObsType = dict[str, Any]


class MultiGridEnv(gym.Env, RandomMixin, ABC):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        agents: int = 1,
        width: int = 20,
        height: int = 20,
        max_steps: int = 100,
        agent_view_size: int = 7,
        highlight: bool = False,
        see_through_walls: bool = False,
        joint_reward: bool = False,
        tile_size=TILE_PIXELS,
        screen_size: int | tuple[int, int] | None = None,
        render_mode: Literal["human", "render_mode"] = "human",
        success_termination_mode: Literal["all", "any"] = "all",
        failure_termination_mode: Literal["all", "any"] = "any",
    ):
        gym.Env.__init__(self)
        ABC.__init__(self)
        RandomMixin.__init__(self, self.np_random)
        assert agents > 0, "Number of agents must be greater than 0"
        self._num_agents = agents
        self._width = width
        self._height = height
        self._highlight = highlight
        self._joint_reward = joint_reward
        self._tile_size = tile_size
        self._render_size = None
        self._window = None
        self._clock = None
        self._step_count = 0
        self._max_steps = max_steps
        self.render_mode = render_mode
        self._success_termination_mode = success_termination_mode
        self._failure_termination_mode = failure_termination_mode

        if screen_size is None:
            screen_size = (width * tile_size, height * tile_size)
        elif isinstance(screen_size, int):
            screen_size = (screen_size, screen_size)
            tile_size = min(screen_size) // max(width, height)
        self._screen_size = screen_size
        assert isinstance(screen_size, tuple)

        self._agent_states = AgentState(agents)
        self.agents: list[Agent] = []
        for i in range(self._num_agents):
            agent = Agent(i, agent_view_size, see_through_walls)
            self.agents.append(agent)
        self.grid = Grid(width, height)
        if not hasattr(self, "grid"):
            self.grid = Grid(width, height)

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, dict[str, Any]],
    ]:
        super().reset(seed=seed, **kwargs)

        # Reset agents
        self.agent_states = AgentState(self._num_agents)
        for agent in self.agents:
            agent.state = self._agent_states[agent.index]
            agent.reset()
            agent.state.pos = self.place_agent(agent)

        self._gen_grid(self._width, self._height)

        # These fields should be defined by _gen_grid
        assert np.all([self.grid.in_bounds(self._agent_states.pos)])

        for agent in self.agents:
            start_cell = self.grid.get(agent.state.pos)
            assert start_cell is None or start_cell.can_overlap()

        self._step_count = 0

        observations = self._gen_obs()

        if self.render_mode == "human":
            self.render()

        return observations, defaultdict(dict)

    def step(
        self, actions: dict[AgentID, Action]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, SupportsFloat],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        self._step_count += 1
        rewards = self._handle_actions(actions)

        observations: dict[AgentID, ObsType] = self._gen_obs()
        terminations: dict[AgentID, bool] = {
            str(agent_id): self._agent_states[agent_id].terminated
            for agent_id in range(self._num_agents)
        }
        truncated: bool = self._step_count >= self._max_steps
        truncations: dict[AgentID, bool] = {
            str(agent_id): truncated
            for agent_id, _ in enumerate(repeat(truncated, self._num_agents))
        }

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, defaultdict(dict)

    def render(self):
        img = self._get_frame(self._highlight, self._tile_size)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            screen_size = tuple(map(int, self._screen_size))
            if self._render_size is None:
                self._render_size = img.shape[2]
            if self._window is None:
                pg.init()
                pg.display.init()
                pg.display.set_caption("MultiGrid")
                self._window = pg.display.set_mode(screen_size)
            if self._clock is None:
                self.clock = pg.time.Clock()
            surf = pg.surfarray.make_surface(img)
            bg = pg.Surface(screen_size)
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (0, 0))
            self._window.blit(bg, (0, 0))
            pg.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pg.display.flip()
        elif self.render_mode == "rgb_array":
            return img
        else:
            raise ValueError("Invalid render mode", self.render_mode)

    @property
    def observation_space(self) -> spaces.Dict:
        """
        Returns
        -------
        spaces.Dict[AgentID, spaces.Space]
            A dictionary of observation spaces for each agent
        """
        return spaces.Dict(
            {agent.index: agent.observation_space for agent in self.agents}
        )

    @property
    def action_space(self) -> spaces.Dict:
        """
        Returns
        -------
        spaces.Dict[AgentID, spaces.Space]
            A dictionary of action spaces for each agent
        """
        return spaces.Dict({agent.index: agent.action_space for agent in self.agents})

    def _handle_actions(
        self, actions: dict[AgentID, Action]
    ) -> dict[AgentID, SupportsFloat]:
        rewards = {agent_index: 0 for agent_index in range(self._num_agents)}

        # TODO: Randomize order

        for i in range(self._num_agents):
            if i not in actions:
                continue

            agent, action = self.agents[i], actions[i]

            if agent.state.terminated:
                continue

            # Rotate left
            if action == Action.left:
                agent.state.dir = (agent.state.dir - 1) % 4

            # Rotate right
            elif action == Action.right:
                agent.state.dir = (agent.state.dir + 1) % 4

            # Move forward
            elif action == Action.forward:
                fwd_pos = agent.front_pos
                if not self.grid.in_bounds(fwd_pos):
                    continue

                fwd_obj = self.grid.get(fwd_pos)

                if fwd_obj is not None and not fwd_obj.can_overlap():
                    continue

                agent_present = np.array(self._agent_states.pos == fwd_pos).any()
                if agent_present:
                    continue

                agent.state.pos = fwd_pos
                if fwd_obj is not None:
                    if fwd_obj.type == WorldObjectType.goal:
                        self.on_success(agent, rewards, {})
                break

            elif action == Action.pickup:
                if agent.state.carrying is not None:
                    break

                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(fwd_pos)

                if fwd_obj is None or not fwd_obj.can_pickup():
                    break

                agent.state.carrying = fwd_obj
                self.grid.set(fwd_pos, None)

            elif action == Action.drop:
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(fwd_pos)

                if agent.state.carrying is not None and fwd_obj is None:
                    agent_present = np.array(self._agent_states.pos == fwd_pos).any()
                    if not agent_present:
                        self.grid.set(fwd_pos, agent.state.carrying)
                        agent.state.carrying.cur_pos = fwd_pos
                        agent.state.carrying = None

            elif action == Action.toggle:
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(fwd_pos)
                if fwd_obj is not None:
                    fwd_obj.toggle(self, agent, fwd_pos)

            elif action == Action.done:
                pass
            else:
                raise ValueError(f"Invalid action: {action}")

        return rewards

    @abstractmethod
    def _gen_grid(self, width: int, height: int):
        raise NotImplementedError

    def _get_frame(self, highlight: bool, tile_size: int) -> np.ndarray:
        return self._get_full_render(highlight, tile_size)

    def _gen_obs(self) -> dict[AgentID, ObsType]:
        direction = self._agent_states.dir
        image = gen_obs_grid_encoding(
            self.grid.state,
            self._agent_states,
            self.agents[0].view_size,
            self.agents[0].see_through_walls,
        )
        observations = {}
        for i in range(self._num_agents):
            observations[i] = {
                "image": image[i],
                "direction": direction[i],
            }

        return observations

    def _get_full_render(self, highlight: bool, tile_size: int) -> np.ndarray:
        obs_shape = self.agents[0].observation_space["image"].shape[:-1]
        vis_mask = np.zeros((self._num_agents, *obs_shape), dtype=bool)
        for i, agent_obs in self._gen_obs().items():
            vis_mask[i] = (
                agent_obs["image"][..., 0] != WorldObjectType.unseen.to_index()
            )

        highlight_mask = np.zeros((self._width, self._height), dtype=bool)

        for agent in self.agents:
            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = agent.state.dir.to_vec()
            r_vec = np.array((f_vec[1], -f_vec[0]))
            top_left = (
                agent.state.pos()
                + f_vec * (agent.view_size - 1)
                - r_vec * (agent.view_size // 2)
            )

            # For each cell in the visability mask
            for vis_j in range(agent.view_size):
                for vis_i in range(agent.view_size):
                    if not vis_mask[agent.index][vis_j, vis_i]:
                        pass
                        # continue
                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_i) + (r_vec * vis_j)
                    # If the cell is within the grid bounds
                    if 0 <= abs_i < self._width and 0 <= abs_j < self._height:
                        highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size, agents=self.agents, highlight_mask=highlight_mask
        )
        return img

    def on_success(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
    ):
        """
        Callback when an agent completes its mission
        """
        if self._success_termination_mode == "any":
            self._agent_states.terminated = True  # Terminate all agents
            for i in range(self._num_agents):
                terminations[str(i)] = True
        else:
            agent.state.terminated = True  # Terminate only the agent
            terminations[str(agent.index)] = True

        if self._joint_reward:
            for i in range(self._num_agents):
                rewards[str(i)] = self._reward()  # Reward all agents
        else:
            rewards[str(agent.index)] = self._reward()

    def _on_failure(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
    ):
        if self._failure_termination_mode == "any":
            self._agent_states.terminated = True
            for i in range(self._num_agents):
                terminations[str(i)] = True
        else:
            agent.state.terminated = True
            terminations[str(agent.index)] = True

    def _is_done(self) -> bool:
        truncated = self._step_count >= self._max_steps
        return truncated or all(self._agent_states.terminated)

    def _reward(self) -> float:
        return 1.0 - 0.9 * (self._step_count / self._max_steps)

    def _place_object(
        self,
        obj: WorldObject | None,
        top: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
        reject_fn: Callable[["MultiGridEnv", tuple[int, int]], bool] | None = None,
        max_tries=math.inf,
    ) -> Position:
        """
        Place an object at an empty position in the grid.

        Parameters
        ----------
        obj: WorldObj
            Object to place in the grid
        top: tuple[int, int]
            Top-left position of the rectangular area where to place the object
        size: tuple[int, int]
            Width and height of the rectangular area where to place the object
        reject_fn: Callable(env, pos) -> bool
            Function to filter out potential positions
        max_tries: int
            Maximum number of attempts to place the object
        """
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = Position(
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # Don't place the object on top of another object
            if self.grid.get(pos) is not None:
                continue

            # Don't place the object where agents are
            if np.array(self.agent_states.pos == pos).any():
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos()):
                continue

            break

        self.grid.set(pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj: WorldObject, pos: Position):
        """
        Put an object at a specific position in the grid.
        """
        self.grid.set(pos, obj)
        obj.init_pos = pos
        obj.cur_pos = pos

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
