import math
from typing import Dict, List, Literal, SupportsFloat, Tuple

import numpy as np

from multiworld.base import MultiWorldEnv
from multiworld.core.position import Position
from multiworld.swarm.core.action import Action
from multiworld.swarm.core.agent import Agent, AgentState
from multiworld.swarm.core.constants import OBJECT_SIZE, WorldObjectType
from multiworld.swarm.core.world import World
from multiworld.swarm.utils.observation import gen_obs_grid_encoding
from multiworld.utils.typing import AgentID, ObsType
from utils.common.callbacks import RenderingCallback, empty_rendering_callback


class SwarmEnv(MultiWorldEnv):
    def __init__(
        self,
        agents: int = 1,
        width: int = 1000,
        height: int = 1000,
        max_steps: int = 100,
        agent_view_size: int = 101,
        observations: int = 10,
        see_through_walls: bool = False,
        joint_reward: bool = False,
        team_reward: bool = False,
        object_size: int = OBJECT_SIZE,
        screen_size: Tuple[int, int] | None = (1000, 1000),
        render_mode: Literal["human", "rgb_array"] = "human",
        rendering_callback: RenderingCallback = empty_rendering_callback,
        success_termination_mode: Literal["all", "any"] = "all",
        failure_termination_mode: Literal["all", "any"] = "any",
        continuous_action_space: bool = False,
    ):
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
            success_termination_mode,
            failure_termination_mode,
        )
        self._object_size = object_size

        self._agent_states = AgentState(agents)
        self._agents: List[Agent] = []
        for i in range(self._num_agents):
            agent = Agent(
                i,
                observations,
                agent_view_size,
                see_through_walls,
                continuous_action_space,
            )
            self._agents.append(agent)
        self._world = World(width, height, object_size)

        self._continuous_action_space = continuous_action_space

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
            agent.state.dir = self._rand_int(0, 360)

        return pos

    @property
    def agents(self) -> List[Agent]:
        return self._agents

    @property
    def agent_states(self) -> AgentState:
        return self._agent_states

    @property
    def world(self) -> World:
        return self._world

    def _execute_action(
        self, agent: Agent, action: Action | int, rewards: Dict[AgentID, SupportsFloat]
    ) -> None:
        if agent.state.terminated:
            return

        def move_forward():
            fwd_pos = agent.front_pos
            if not self.world.in_bounds(fwd_pos):
                x = fwd_pos.x % (self._width - self.world.object_size)
                y = fwd_pos.y % (self._height - self.world.object_size)
                fwd_pos = Position(x, y)

            fwd_obj = self.world.get(fwd_pos)

            if fwd_obj is not None and not fwd_obj.can_overlap():
                return

            agent_present = np.array(self._agent_states.pos == fwd_pos).any()
            if agent_present:
                return

            agent.state.pos = fwd_pos
            if fwd_obj is not None:
                if fwd_obj.type == WorldObjectType.goal:
                    self.on_success(agent, rewards, {})

        if self._continuous_action_space:
            agent.state.dir = (agent.dir + action % 360) % 360
        else:
            if action == Action.left45:
                agent.state.dir = (agent.dir - 45) % 360
            elif action == Action.left90:
                agent.state.dir = (agent.dir - 90) % 360
            elif action == Action.forward:
                pass
            elif action == Action.right45:
                agent.state.dir = (agent.dir + 45) % 360
            elif action == Action.right90:
                agent.state.dir = (agent.dir + 90) % 360
            else:
                raise ValueError(f"Invalid action: {action}")

        move_forward()

    def _get_frame(self) -> np.ndarray:
        return self.world.render(
            self._object_size,
            agents=self._agents,
            world_objects=self.world.world_objects.values(),
        )

    def _gen_obs(self) -> Dict[AgentID, ObsType]:
        directions = self._agent_states.dir
        """
        world_objects = gen_obs_encoding(
            self.world._world_objects, self._agent_states, self._agents[0].view_size
        )
        """
        obs = gen_obs_grid_encoding(
            self._agent_states,
            self._agents[0].view_size,
            self._agents[0].observations,
            world_size=(self._width, self._height),
        )
        observations = {}
        for i in range(self._num_agents):
            observations[i] = {
                "observation": obs[i],
                "direction": directions[i],
            }
        return observations

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
