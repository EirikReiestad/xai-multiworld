from typing import Any, Dict, SupportsFloat, Tuple

import numpy as np

from multiworld.base import MultiWorldEnv
from multiworld.core.action import Action
from multiworld.core.agent import Agent, AgentState
from multiworld.core.world import World
from multiworld.utils.typing import AgentID, ObsType

AGENT_POS_IDX = AgentState.POS


class FlockEnv(MultiWorldEnv):
    def _gen_world(self, width: int, height: int):
        self.world = World(width, height, self._object_size)

        for agent in self.agents:
            position = self.world.get_empty_position(self.np_random)
            agent.state.pos = position

    def step(
        self, actions: Dict[AgentID, Action | int]
    ) -> Tuple[
        Dict[AgentID, ObsType],
        Dict[AgentID, SupportsFloat],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Dict[str, Any]],
    ]:
        observations, step_rewards, terminations, truncations, infos = super().step(
            actions
        )

        def move_forward(agent: Agent):
            fwd_pos = agent.front_pos
            if not self.world.in_bounds(fwd_pos):
                return

            fwd_obj = self.world.get(fwd_pos)

            if fwd_obj is not None and not fwd_obj.can_overlap():
                return

            agent_present = np.array(self._agent_states.pos == fwd_pos).any()
            if agent_present:
                return

            agent.state.pos = fwd_pos

        rewards: Dict[AgentID, SupportsFloat] = {
            agent.index: 0 for agent in self.agents
        }

        for agent, _ in actions.items():
            move_forward(self.agents[agent])
            agents_view_count = (
                self._get_agents_view_count(self.agents[agent].pos(), 20) - 1
            )
            self.add_reward(
                self.agents[agent],
                rewards,
                np.exp(0.2 * agents_view_count) * 0.1 * (agents_view_count > 0),
            )

        rewards = {
            agent.index: float(rewards[agent.index]) + float(step_rewards[agent.index])
            for agent in self.agents
        }

        return observations, rewards, terminations, truncations, infos

    def _get_agents_view_count(self, pos: Agent, radius: int):
        agent_pos = self._agent_states[..., AGENT_POS_IDX]
        diffs = np.abs(agent_pos - pos)
        return len(np.where((diffs[:, 0] <= radius) & (diffs[:, 1] <= radius))[0])
