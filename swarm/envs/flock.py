import math
from typing import Any, Dict, SupportsFloat, Tuple

import numpy as np

from swarm.base import SwarmEnv
from swarm.core.action import Action
from swarm.core.agent import Agent, AgentState
from swarm.core.world import World
from swarm.utils.position import Position
from swarm.utils.typing import AgentID, ObsType

AGENT_POS_IDX = AgentState.POS


class FlockEnv(SwarmEnv):
    def __init__(self, agents: int = 4, predators: int = 1, *args, **kwargs):
        super().__init__(agents=agents + predators, *args, **kwargs)
        self._num_predators = predators
        self._num_active_agents = self._num_agents - predators

    def _gen_world(self, width: int, height: int):
        self.world = World(width, height, self._object_size)

        for agent in self.agents:
            position = self.world.get_empty_position(self.np_random)
            agent.state.pos = position
            agent.color = "white"

        for i in range(self._num_active_agents, self._num_agents):
            self.agents[i].color = "red"

        self._predator_info = []
        self._predator_steps = 100

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

        rewards: Dict[AgentID, SupportsFloat] = {
            agent.index: 0 for agent in self.agents
        }

        for agent, _ in actions.items():
            agents_view_count = (
                self._get_agents_view_count(self.agents[agent].pos(), 20) - 1
            )
            if self.agents[agent].stamina <= 0:
                self.agents[agent].terminated = True
                self.agents[agent].pos = (-1, -1)
                self.add_reward(
                    self.agents[agent], rewards, -self._max_steps / len(self.agents)
                )
            self.add_reward(self.agents[agent], rewards, agents_view_count * 0.1)
            self.agents[agent].stamina = min(
                self.agents[agent].stamina + agents_view_count,
                self._agent_stamina,
            )

        rewards = {
            agent.index: float(rewards[agent.index]) + float(step_rewards[agent.index])
            for agent in self.agents
        }

        for i in range(self._num_predators):
            predator = self.agents[self._num_active_agents + i]
            predator_info = self._predator_info[i]

            if predator_info.get("steps_left") in [None, 0]:
                predator_info["following"] = None
                predator_info["steps_left"] = self._predator_steps

            if predator_info["following"] is None:
                predator_info["following"] = self._find_closest_agent(predator.pos())

            # Make the predator follow the assigned agent
            self._follow_agent(i, predator_info["following"])

        return (
            dict(list(observations.items())[: self._num_active_agents]),
            dict(list(rewards.items())[: self._num_active_agents]),
            dict(list(terminations.items())[: self._num_active_agents]),
            dict(list(terminations.items())[: self._num_active_agents]),
            dict(list(infos.items())[: self._num_active_agents]),
        )

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> Tuple[
        Dict[AgentID, ObsType],
        Dict[AgentID, Dict[str, Any]],
    ]:
        observations, infos = super().reset(seed=seed, **kwargs)
        return dict(list(observations.items())[: self._num_predators]), dict(
            list(infos.items())[: self._num_predators]
        )

    def _follow_agent(self, predator_idx: int, agent_idx: int, rewards: Dict):
        predator = self.agents[self._num_active_agents + predator_idx]
        agent = self.agents[agent_idx]

        dx = agent.pos.x - predator.pos.x
        dy = agent.pos.y - predator.pos.y

        magnitude = (dx**2 + dy**2) ** 0.5
        dx_normalized = dx / magnitude
        dy_normalized = dy / magnitude
        angle = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle)
        new_pos = (
            predator.pos.x + dx_normalized,
            predator.pos.y + dy_normalized,
        )
        self.agents[self._num_active_agents + predator_idx].dir = angle_degrees
        self.agents[self._num_active_agents + predator_idx].pos = new_pos

        for i in range(self._num_active_agents):
            if self.agents[i].pos == new_pos:
                self.agents[i].terminated = True
                self.add_reward(self.agents[i], rewards, -self._max_steps)
                self.agents[i].pos = (-1, -1)

    def _find_closest_agent(self, pos: Tuple[int, int]) -> int:
        closest_agent = None
        closest_distance = None
        for i in range(self._num_active_agents):
            if self.agents[i].terminated is True:
                continue
            distance = np.linalg.norm(np.array(pos) - np.array(self.agents[i]))
            if closest_distance is None or distance < closest_distance:
                closest_agent = i
                closest_distance = distance
        if closest_agent is None:
            raise EnvironmentError(
                "For some reason there are no more agents and the episode is not terminated..."
            )
        return closest_agent

    def _get_agents_view_count(self, pos: Agent, radius: int):
        agent_pos = self._agent_states[..., AGENT_POS_IDX]
        diffs = np.abs(agent_pos - pos)
        return len(np.where((diffs[:, 0] <= radius) & (diffs[:, 1] <= radius))[0])
