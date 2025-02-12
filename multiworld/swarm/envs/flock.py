import math
from typing import Any, Dict, SupportsFloat, Tuple

import numpy as np

from multiworld.core.position import Position
from multiworld.swarm.base import SwarmEnv
from multiworld.swarm.core.action import Action
from multiworld.swarm.core.agent import Agent, AgentState
from multiworld.swarm.core.world import World
from multiworld.swarm.utils.observation import wrapped_distance
from multiworld.utils.typing import AgentID, ObsType

AGENT_POS_IDX = AgentState.POS


class FlockEnv(SwarmEnv):
    def __init__(
        self,
        agents: int = 4,
        predators: int = 10,
        predator_steps: int = 100,
        max_predator_angle_change: int = 45,
        *args,
        **kwargs,
    ):
        super().__init__(agents=agents + predators, *args, **kwargs)
        self._num_predators = predators
        self._predator_steps = predator_steps
        self._max_predator_angle_change = max_predator_angle_change
        self._num_active_agents = self._num_agents - predators

    def _gen_world(self, width: int, height: int):
        self._world = World(width, height, self._object_size)

        for agent in self.agents:
            position = self.world.get_empty_position(self.np_random)
            agent.state.pos = position
            agent.color = "white"

        for i in range(self._num_active_agents, self._num_agents):
            self.agents[i].color = "red"

        self._predator_info = [
            {"following": None, "steps_left": self._predator_steps}
            for _ in range(self._num_predators)
        ]

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
            self.add_reward(
                self.agents[agent],
                rewards,
                1 / self._num_active_agents / self._max_steps,
            )

        for i in range(self._num_predators):
            predator = self.agents[self._num_active_agents + i]
            predator_info = self._predator_info[i]

            if predator_info["steps_left"] == 0:
                predator_info["following"] = None
                predator_info["steps_left"] = self._predator_steps

            if predator_info["following"] is None:
                following = self._find_closest_agent(predator.pos)
                if following is None:
                    break
                predator_info["steps_left"] = self._predator_steps
                predator_info["following"] = following

            predator_info["steps_left"] -= 1
            self._follow_agent(i, predator_info["following"], rewards)

        rewards = {
            agent.index: float(rewards[agent.index]) + float(step_rewards[agent.index])
            for agent in self.agents
        }

        return (
            dict(list(observations.items())[: self._num_active_agents]),
            dict(list(rewards.items())[: self._num_active_agents]),
            dict(list(terminations.items())[: self._num_active_agents]),
            dict(list(truncations.items())[: self._num_active_agents]),
            dict(list(infos.items())[: self._num_active_agents]),
        )

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> Tuple[
        Dict[AgentID, ObsType],
        Dict[AgentID, Dict[str, Any]],
    ]:
        observations, infos = super().reset(seed=seed, **kwargs)
        return dict(list(observations.items())[: self._num_active_agents]), dict(
            list(infos.items())[: self._num_active_agents]
        )

    def _follow_agent(self, predator_idx: int, agent_idx: int, rewards: Dict):
        predator = self.agents[self._num_active_agents + predator_idx]
        agent = self.agents[agent_idx]

        if agent.terminated:
            self._predator_info[predator_idx]["following"] = None

        dx = agent.pos.x - predator.pos.x
        dy = agent.pos.y - predator.pos.y

        dx = (dx + self._width / 2) % self._width - self._width / 2
        dy = (dy + self._height / 2) % self._height - self._height / 2

        target_angle = math.atan2(dy, dx)
        target_angle_degrees = math.degrees(target_angle)

        angle_diff = target_angle_degrees - predator.dir

        angle_diff = max(
            -self._max_predator_angle_change,
            min(angle_diff, self._max_predator_angle_change),
        )

        predator.dir += angle_diff
        fwd_pos = predator.front_pos
        if not self.world.in_bounds(fwd_pos):
            x = fwd_pos.x % (self._width - self.world.object_size)
            y = fwd_pos.y % (self._height - self.world.object_size)
            fwd_pos = Position(x, y)
        predator.pos = fwd_pos

        if agent.pos == predator.pos:
            agent.terminated = True
            agent.pos = (-1, -1)
            self.add_reward(agent, rewards, -1)
            self._predator_info[predator_idx]["steps_left"] = 0

    def _find_closest_agent(self, pos: Position) -> int | None:
        probabilities = []
        total_probability = 0
        p = 2

        for i in range(self._num_active_agents):
            if self.agents[i].terminated is True:
                probabilities.append(0)
                continue

            distance = np.linalg.norm(pos.to_numpy() - self.agents[i].pos.to_numpy())
            distance = wrapped_distance(
                pos.to_numpy(),
                self.agents[i].pos.to_numpy(),
                (self._width, self._height),
            )

            probability = 1 / (distance + 1e-6) ** p
            probabilities.append(probability)
            total_probability += probability

        if total_probability == 0:
            return None

        normalized_probabilities = [p / total_probability for p in probabilities]
        chosen_agent = self._rand_choice(
            range(self._num_active_agents), p=normalized_probabilities
        )
        return chosen_agent

    def _get_agents_view_count(self, pos: Agent, radius: int):
        agent_pos = self._agent_states[..., AGENT_POS_IDX]
        diffs = np.abs(agent_pos - pos)
        return len(np.where((diffs[:, 0] <= radius) & (diffs[:, 1] <= radius))[0])
