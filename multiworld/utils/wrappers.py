from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Literal, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from numpy.typing import NDArray as ndarray

from multiworld.base import MultiWorldEnv
from utils.core.constants import Color
from multiworld.multigrid.base import MultiGridEnv
from multiworld.multigrid.core.constants import Direction, State, WorldObjectType
from multiworld.multigrid.core.world_object import WorldObject
from multiworld.multigrid.utils.decoder import decode_observation
from multiworld.utils.advanced_typing import Action
from multiworld.utils.serialization import (
    deserialize_observation,
    serialize_observation,
)
from multiworld.utils.typing import AgentID, ObsType


@dataclass
class Observations:
    observations: Dict[AgentID, ObsType]
    actions: Dict[AgentID, int]
    rewards: Dict[AgentID, SupportsFloat]
    terminations: Dict[AgentID, bool]
    truncations: Dict[AgentID, bool]
    infos: Dict[AgentID, Dict[str, Any]]

    def serialize(self):
        return {
            "observations": {
                agent_id: serialize_observation(obs)
                for agent_id, obs in self.observations.items()
            },
            "actions": self.actions,
            "rewards": self.rewards,
            "terminations": self.terminations,
            "truncations": self.truncations,
            "infos": {
                agent_id: {
                    key: str(value)
                    if isinstance(value, (int, float, str))
                    else str(value)
                    for key, value in info.items()
                }
                for agent_id, info in self.infos.items()
            },
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> "Observations":
        return Observations(
            observations={
                agent_id: deserialize_observation(obs)
                for agent_id, obs in data["observations"].items()
            },
            actions={
                agent_id: int(action) for agent_id, action in data["actions"].items()
            },
            rewards={
                agent_id: float(reward) for agent_id, reward in data["rewards"].items()
            },
            terminations={
                agent_id: bool(termination)
                for agent_id, termination in data["terminations"].items()
            },
            truncations={
                agent_id: bool(truncation)
                for agent_id, truncation in data["truncations"].items()
            },
            infos={
                agent_id: {
                    key: value if isinstance(value, (int, float, str)) else value
                    for key, value in info.items()
                }
                for agent_id, info in data["infos"].items()
            },
        )


class ObservationCollectorWrapper(gym.Wrapper):
    def __init__(
        self,
        env: MultiWorldEnv,
        observations: int = 1000,
        sample_rate: float = 1.0,
        directory: str = os.path.join("assets", "observations"),
        filename: str = "observations",
        verbose: bool = False,
    ) -> None:
        super().__init__(env)
        self.env = env
        self._sample_rate = sample_rate
        self._filepath = os.path.join(directory, filename + ".json")
        self._observations = observations

        self._rollouts: List[Observations] = []
        self._verbose = verbose

    def step(
        self,
        actions: Dict[AgentID, int],
    ):
        observations, rewards, terminations, truncations, infos = super().step(actions)

        if np.random.rand() <= self._sample_rate:
            self._rollouts.append(
                Observations(
                    observations, actions, rewards, terminations, truncations, infos
                )
            )

        if len(self._rollouts) % (self._observations / 10) == 0 and self._verbose:
            logging.info(
                f"Collcted {len(self._rollouts)} / {self._observations} observations"
            )

        if len(self._rollouts) == self._observations:
            if self._verbose:
                logging.info(f"Saving rollouts to file {self._filepath}...")
            self._save_rollouts()
            sys.exit()

        return observations, rewards, terminations, truncations, infos

    def _save_rollouts(self):
        data = [rollout.serialize() for rollout in self._rollouts]

        with open(self._filepath, "w") as f:
            json.dump(data, f, indent=4)


class ConceptObsWrapper(gym.Wrapper):
    """
    Collect observations for a concept learning task.
    """

    def __init__(
        self,
        env: MultiGridEnv,  #  | SwarmEnv,
        concept_checks: Callable,
        observations: int = 1000,
        concepts: List[str] | None = None,
        method: Literal["random", "policy"] = "policy",
        save_dir: str = "assets/concepts",
        result_save_dir: str = "assets/results",
    ):
        super().__init__(env)

        self._num_observations = observations
        self._save_dir = save_dir
        self._result_save_dir = result_save_dir
        os.makedirs(self._save_dir, exist_ok=True)

        self._concepts: Dict[str, List[ObsType]] = defaultdict(list)
        self._concept_checks = concept_checks(concepts)
        keys = list(self._concept_checks.keys()) + [
            "negative_" + key for key in self._concept_checks.keys()
        ]
        self._concepts_filled = {key: False for key in keys}

        assert len(self._concept_checks) != 0, f"No concepts to check, {concepts}"

        self._decoder = partial(decode_observation, preprocessing=env._preprocessing)

        self._method = method
        self._step_count = 0
        self._concepts_added = 0
        self._previous_concepts_added = 0
        self._sample_efficiency = {key: 0 for key in self._concept_checks.keys()}
        self._timeout = 20

    def step(
        self, actions: Dict[AgentID, Action | int]
    ) -> Tuple[
        Dict[AgentID, ObsType],
        Dict[AgentID, SupportsFloat],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Dict[str, Any]],
    ]:
        if self._method == "random":
            super().reset()

        if self._step_count % self._num_observations == 0:
            if (
                self._previous_concepts_added == self._concepts_added
                and self._step_count > 0
            ):
                self._timeout -= 1
                if self._timeout == 0:
                    info_str = "\n"
                    keys = list(self._concept_checks.keys()) + [
                        "negative_" + key for key in self._concept_checks.keys()
                    ]
                    for key in keys:
                        info_str += f"{key} - {len(self._concepts.get(key) or [])}\n"
                    self._write_concepts()
                    logging.warning("Can not generate all concepts." + info_str)
                    sys.exit()
            else:
                logging.info(f"Step {self._step_count}")
                logging.info(
                    f"Number of concepts filled: {self._concepts_added} / {self._num_observations * len(self._concept_checks) * 2}"
                )
            self._previous_concepts_added = self._concepts_added

        observations, rewards, terminations, truncations, info = super().step(actions)

        for concept, check_fn in self._concept_checks.items():
            negative_concept = "negative_" + concept
            if all(self._concepts_filled.values()):
                self._write_concepts()
                sys.exit()

            if (
                self._concepts_filled[concept]
                and self._concepts_filled[negative_concept]
            ):
                continue

            for agent_id, obs in observations.items():
                decoded_obs = self._decoder(obs.copy())
                if not self._concepts_filled[concept]:
                    self._sample_efficiency[concept] += 1

                if not check_fn(decoded_obs):
                    if self._concepts_filled[negative_concept]:
                        continue
                    rand_float = np.random.uniform()
                    if rand_float < 0.2:
                        self._concepts[negative_concept].append(obs)
                        self._concepts_added += 1
                    if len(self._concepts[negative_concept]) >= self._num_observations:
                        self._concepts_filled[negative_concept] = True
                        break
                    continue

                if self._concepts_filled[concept]:
                    continue

                self._concepts_added += 1
                self._concepts[concept].append(obs)

                if len(self._concepts[concept]) >= self._num_observations:
                    self._concepts_filled[concept] = True
                    break

        self._step_count += 1

        return observations, rewards, terminations, truncations, info

    def _write_concepts(self) -> None:
        logging.info("Writing concept observations to disk...")
        for concept, observations in self._concepts.items():
            filename = f"{concept}.json"
            path = os.path.join(self._save_dir, filename)

            with open(path, "w") as f:
                json.dump(observations, f, indent=4, cls=self.encoder)

        path = os.path.join(self._result_save_dir, "sample_efficiency.json")
        results = {}
        for concept in self._sample_efficiency:
            normalized = self._num_observations / self._sample_efficiency[concept]
            results[concept] = {
                "num_observations": self._num_observations,
                "num_samples": self._sample_efficiency[concept],
                "normalized": normalized,
            }

        with open(path, "w") as f:
            json.dump(results, f, indent=4)


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding instead of agent view.

    Examples
    --------
    >>> import gymnasium as gym
    >>> import multigrid.envs
    >>> env = gym.make('MultiGrid-Empty-16x16-v0')
    >>> obs, _ = env.reset()
    >>> obs[0]['observation'].shape
    (7, 7, 3)

    >>> from multigrid.wrappers import FullyObsWrapper
    >>> env = FullyObsWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['observation'].shape
    (16, 16, 3)
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)

        # Update agent observation spaces
        for agent in self.env.agents:
            agent.observation_space["observation"] = spaces.Box(
                low=0,
                high=255,
                shape=(env.height, env.width, WorldObject.dim),
                dtype=np.int32,
            )

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        img = self.env.grid.encode()
        for agent in self.env.agents:
            img[agent.state.pos] = agent.encode()

        for agent_id in obs:
            obs[agent_id]["observation"] = img

        return obs


class ImgObsWrapper(ObservationWrapper):
    """
    Use the observation as the only observation output for each agent.

    Examples
    --------
    >>> import gymnasium as gym
    >>> import multigrid.envs
    >>> env = gym.make('MultiGrid-Empty-8x8-v0')
    >>> obs, _ = env.reset()
    >>> obs[0].keys()
    dict_keys(['observation', 'direction', 'mission'])

    >>> from multigrid.wrappers import ImgObsWrapper
    >>> env = ImgObsWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs.shape
    (7, 7, 3)
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)

        # Update agent observation spaces
        for agent in self.env.agents:
            agent.observation_space = agent.observation_space["observation"]
            agent.observation_space.dtype = np.uint8

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        for agent_id in obs:
            obs[agent_id] = obs[agent_id]["observation"].astype(np.uint8)

        return obs


class OneHotObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Examples
    --------
    >>> import gymnasium as gym
    >>> import multigrid.envs
    >>> env = gym.make('MultiGrid-Empty-5x5-v0')
    >>> obs, _ = env.reset()
    >>> obs[0]['observation'][0, :, :]
    array([[2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0]])

    >>> from multigrid.wrappers import OneHotObsWrapper
    >>> env = OneHotObsWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['observation'][0, :, :]
    array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
            dtype=uint8)
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.dim_sizes = np.array(
            [len(WorldObjectType), len(Color), max(len(State), len(Direction))]
        )

        # Update agent observation spaces
        dim = sum(self.dim_sizes)
        for agent in self.env.agents:
            view_height, view_width, _ = agent.observation_space["observation"].shape
            agent.observation_space["observation"] = spaces.Box(
                low=0, high=1, shape=(view_height, view_width, dim), dtype=np.uint8
            )

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        for agent_id in obs:
            obs[agent_id]["observation"] = self.one_hot(
                obs[agent_id]["observation"], self.dim_sizes
            )

        return obs

    @staticmethod
    def one_hot(x: ndarray[np.int], dim_sizes: ndarray[np.int]) -> ndarray[np.uint8]:
        """
        Return a one-hot encoding of a 3D integer array,
        where each 2D slice is encoded separately.

        Parameters
        ----------
        x : ndarray[int] of shape (view_height, view_width, dim)
            3D array of integers to be one-hot encoded
        dim_sizes : ndarray[int] of shape (dim,)
            Number of possible values for each dimension

        Returns
        -------
        out : ndarray[uint8] of shape (view_height, view_width, sum(dim_sizes))
            One-hot encoding

        :meta private:
        """
        out = np.zeros((x.shape[0], x.shape[1], sum(dim_sizes)), dtype=np.uint8)

        dim_offset = 0
        for d in range(len(dim_sizes)):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k = dim_offset + x[i, j, d]
                    out[i, j, k] = 1

            dim_offset += dim_sizes[d]

        return out


class SingleAgentWrapper(gym.Wrapper):
    """
    Wrapper to convert a multi-agent environment into a
    single-agent environment.

    Examples
    --------
    >>> import gymnasium as gym
    >>> import multigrid.envs
    >>> env = gym.make('MultiGrid-Empty-5x5-v0')
    >>> obs, _ = env.reset()
    >>> obs[0].keys()
    dict_keys(['observation', 'direction', 'mission'])

    >>> from multigrid.wrappers import SingleAgentWrapper
    >>> env = SingleAgentWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs.keys()
    dict_keys(['observation', 'direction', 'mission'])
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.observation_space = env.agents[0].observation_space
        self.action_space = env.agents[0].action_space

    def reset(self, *args, **kwargs):
        """
        :meta private:
        """
        result = super().reset(*args, **kwargs)
        return_value = tuple([list(result[0].values())[0], result[1]])
        return return_value

    def step(self, action):
        """
        :meta private:
        """
        result = super().step({0: action})
        return tuple(item[0] for item in result)
