from abc import ABC
from collections import defaultdict
from itertools import repeat
from typing import Any, Literal, SupportsFloat

import gymnasium as gym
import numpy as np
import pygame as pg
from gymnasium import spaces

from multigrid.core.action import Action
from multigrid.core.agent import Agent, AgentState
from multigrid.core.constants import TILE_PIXELS, Type
from multigrid.core.grid import Grid
from multigrid.utils.observation import gen_obs_grid_encoding

AgentID = str
ObsType = dict[str, Any]


class MultiAgentEnv(gym.Env, ABC):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        agents: int = 1,
        width: int = 20,
        height: int = 10,
        max_steps: int = 100,
        highlight: bool = False,
        tile_size=TILE_PIXELS,
        screen_size: int | tuple[int, int] | None = None,
        render_mode: Literal["human", "render_mode"] = "human",
    ):
        gym.Env.__init__(self)
        ABC.__init__(self)
        assert agents > 0, "Number of agents must be greater than 0"
        self._num_agents = agents
        self._width = width
        self._height = height
        self._highlight = highlight
        self._tile_size = tile_size
        self._render_size = None
        self._window = None
        self._clock = None
        self._step_count = 0
        self._max_steps = max_steps
        self.render_mode = render_mode

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
            agent = Agent(i)
            self.agents.append(agent)
        self.grid = Grid(width, height)
        if not hasattr(self, "grid"):
            self.grid = Grid(width, height)

    def reset(self, seed=None):
        self._gen_grid(self._width, self._height)
        pass

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
                fwd_obj = self.grid.get(*fwd_pos)

                if fwd_obj is None or fwd_obj.can_overlap():
                    agent_present = np.bitwise_and.reduce(
                        self._agent_states.pos == fwd_pos, axis=1
                    ).any()
                    if agent_present:
                        continue

                agent.state.pos = fwd_pos
                if fwd_obj is not None:
                    if fwd_obj.type == Type.goal:
                        self._on_success(agent, rewards, {})

            elif action == Action.pickup:
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(*fwd_pos)

                if fwd_obj is not None and fwd_obj.can_pickup():
                    if agent.state.carrying is None:
                        agent.state.carrying = fwd_obj
                        self.grid.set(*fwd_pos, None)

            elif action == Action.drop:
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(*fwd_pos)

                if agent.state.carrying is not None and fwd_obj is None:
                    agent_present = np.bitwise_and.reduce(
                        self._agent_states.pos == fwd_pos, axis=1
                    ).any()
                    if not agent_present:
                        self.grid.set(*fwd_pos, agent.state.carrying)
                        agent.state.carrying.cur_pos = fwd_pos
                        agent.state.carrying = None

            elif action == Action.toggle:
                fwd_pos = agent.front_pos
                fwd_obj = self.grid.get(*fwd_pos)
                if fwd_obj is not None:
                    fwd_obj.toggle(self, agent, fwd_pos)

            elif action == Action.done:
                pass
            else:
                raise ValueError(f"Invalid action: {action}")

        return rewards

    def _gen_grid(self, width: int, height: int):
        raise NotImplementedError

    def _get_frame(self, highlight: bool, tile_size: int) -> np.ndarray:
        return self._get_full_render(highlight, tile_size)

    def _gen_obs(self) -> dict[AgentID, ObsType]:
        direction = self._agent_states.dir
        image = gen_obs_grid_encoding(
            self.grid.state, self._agent_states, self.agents[0].view_size
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
            vis_mask[i] = agent_obs["image"][..., 0] != Type.unseen.to_index()

        highlight_mask = np.zeros((self._width, self._height), dtype=bool)

        for agent in self.agents:
            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = agent.state.dir.to_vec()
            r_vec = np.array((f_vec[1], -f_vec[0]))
            top_left = (
                agent.state.pos
                + f_vec * (agent.view_size - 1)
                - r_vec * (agent.view_size // 2)
            )

            # For each cell in the visability mask
            for vis_j in range(agent.view_size):
                for vis_i in range(agent.view_size):
                    if not vis_mask[agent.index][vis_j, vis_i]:
                        continue
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
