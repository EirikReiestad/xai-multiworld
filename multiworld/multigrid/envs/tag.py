from typing import Any, SupportsFloat, Dict, Tuple

import numpy as np

from multiworld.multigrid.base import MultiGridEnv
from multiworld.multigrid.core.action import Action
from multiworld.multigrid.core.grid import Grid
from multiworld.utils.typing import AgentID, ObsType


class TagEnv(MultiGridEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._runner_color = "green"
        self._tagger_color = "red"

    def _gen_world(self, width: int, height: int):
        self.grid = Grid(width, height)

        placeable_positions = self.grid.get_empty_positions(len(self.agents))
        for agent, pos in zip(self.agents, placeable_positions):
            agent.state.pos = pos
            agent.color = self._runner_color
        self.agents[0].color = self._tagger_color

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> tuple[
        Dict[AgentID, ObsType],
        Dict[AgentID, Dict[str, Any]],
    ]:
        observations, info = super().reset()
        self._success_move_box = 0
        return observations, info

    def step(
        self, actions: Dict[AgentID, Action | int]
    ) -> Tuple[
        Dict[AgentID, ObsType],
        Dict[AgentID, SupportsFloat],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Dict[str, Any]],
    ]:
        rewards: Dict[AgentID, SupportsFloat] = {
            agent.index: 0 for agent in self.agents
        }
        for agent in self.agents:
            if agent.color == self._runner_color:
                continue
            if agent.color != self._tagger_color:
                raise ValueError(f"Invalid color {agent.color}")

            fwd_pos = agent.front_pos

            agent_present = np.where(np.array(self._agent_states.pos == fwd_pos))[0]
            assert len(agent_present) <= 1

            if len(agent_present) == 0:
                continue

            agent.color = self._runner_color
            self.agents[agent_present[0]].color = self._tagger_color
            break

        for agent in self.agents:
            if agent.color == self._runner_color:
                self.add_reward(agent, rewards, 0.1)
            if agent.color == self._tagger_color:
                self.add_reward(agent, rewards, -0.1)

        observations, step_rewards, terminations, truncations, info = super().step(
            actions
        )

        if self._step_count == self._max_steps:
            for agent in self.agents:
                if agent.color == self._runner_color:
                    self.add_reward(agent, rewards, 1)
                if agent.color == self._tagger_color:
                    self.add_reward(agent, rewards, -1)

        rewards = {
            agent.index: float(rewards[agent.index]) + float(step_rewards[agent.index])
            for agent in self.agents
        }

        return observations, rewards, terminations, truncations, info
