from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Literal, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from numpy.typing import NDArray as ndarray

from multigrid.base import AgentID, MultiGridEnv, ObsType
from multigrid.core.action import Action
from multigrid.core.concept import get_concept_checks
from multigrid.core.constants import Color, Direction, State, WorldObjectType
from multigrid.core.world_object import WorldObject
from utils.common.numpy_encoder import NumpyEncoder


class ConceptObsWrapper(gym.Wrapper):
    """
    Collect observations for a concept learning task.
    """

    def __init__(
        self,
        env: MultiGridEnv,
        observations: int = 1000,
        concepts: List[str] | None = None,
        method: Literal["random", "policy"] = "policy",
        save_dir: str = "assets/concepts",
    ):
        super().__init__(env)

        self._num_observations = observations
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)

        self._concepts: Dict[str, List[ObsType]] = defaultdict(list)
        self._concepts_filled = defaultdict(lambda: False)
        self._concepts_filled["flag"] = True  # Only write once
        self._concept_checks = get_concept_checks(concepts)

        self._method = method

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

        observations, rewards, terminations, truncations, info = super().step(actions)

        for concept, check_fn in self._concept_checks.items():
            if self._concepts_filled[concept]:
                continue
            for agent_id, obs in observations.items():
                if not check_fn(obs):
                    continue
                self._concepts[concept].append(obs)

                if len(self._concepts[concept]) >= self._num_observations:
                    self._concepts_filled[concept] = True

                if all(self._concepts_filled.values()):
                    self._write_concepts()
                    self._concepts_filled["flag"] = False
                    sys.exit()

        return observations, rewards, terminations, truncations, info

    def _write_concepts(self) -> None:
        logging.info("Writing concept observations to disk...")
        for concept, observations in self._concepts.items():
            filename = f"{concept}.json"
            path = os.path.join(self._save_dir, filename)

            with open(path, "w") as f:
                json.dump(observations, f, indent=4, cls=NumpyEncoder)


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding instead of agent view.

    Examples
    --------
    >>> import gymnasium as gym
    >>> import multigrid.envs
    >>> env = gym.make('MultiGrid-Empty-16x16-v0')
    >>> obs, _ = env.reset()
    >>> obs[0]['image'].shape
    (7, 7, 3)

    >>> from multigrid.wrappers import FullyObsWrapper
    >>> env = FullyObsWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['image'].shape
    (16, 16, 3)
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)

        # Update agent observation spaces
        for agent in self.env.agents:
            agent.observation_space["image"] = spaces.Box(
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
            obs[agent_id]["image"] = img

        return obs


class ImgObsWrapper(ObservationWrapper):
    """
    Use the image as the only observation output for each agent.

    Examples
    --------
    >>> import gymnasium as gym
    >>> import multigrid.envs
    >>> env = gym.make('MultiGrid-Empty-8x8-v0')
    >>> obs, _ = env.reset()
    >>> obs[0].keys()
    dict_keys(['image', 'direction', 'mission'])

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
            agent.observation_space = agent.observation_space["image"]
            agent.observation_space.dtype = np.uint8

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        for agent_id in obs:
            obs[agent_id] = obs[agent_id]["image"].astype(np.uint8)

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
    >>> obs[0]['image'][0, :, :]
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
    >>> obs[0]['image'][0, :, :]
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
            view_height, view_width, _ = agent.observation_space["image"].shape
            agent.observation_space["image"] = spaces.Box(
                low=0, high=1, shape=(view_height, view_width, dim), dtype=np.uint8
            )

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        for agent_id in obs:
            obs[agent_id]["image"] = self.one_hot(
                obs[agent_id]["image"], self.dim_sizes
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
    dict_keys(['image', 'direction', 'mission'])

    >>> from multigrid.wrappers import SingleAgentWrapper
    >>> env = SingleAgentWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs.keys()
    dict_keys(['image', 'direction', 'mission'])
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
        return tuple(item[0] for item in result)

    def step(self, action):
        """
        :meta private:
        """
        result = super().step({0: action})
        return tuple(item[0] for item in result)
