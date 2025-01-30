import logging
import math

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray as ndarray

from multiworld.core.constants import Color
from multiworld.core.position import Position
from multiworld.swarm.core.action import Action
from multiworld.swarm.core.constants import WorldObjectType
from multiworld.swarm.utils.misc import front_pos
from multiworld.utils.misc import PropertyAlias
from multiworld.utils.rendering import fill_coords, point_in_triangle, rotate_fn


class Agent:
    def __init__(
        self,
        index: int,
        observations: int,
        view_size: int = 7,
        see_through_walls: bool = False,
        continuous: bool = False,
    ):
        self.index = index
        assert view_size % 2 == 1, "View size must be odd for agent observation."
        assert view_size > 1, "View size must be greater than 1 for agent observation."
        self.view_size = view_size
        self.see_through_walls = see_through_walls
        self.state = AgentState()

        self.observations = observations
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, observations, AgentState.encode_dim),
                    dtype=np.int_,
                ),
                "other": spaces.Discrete(1),
            }
        )
        if continuous:
            self.action_space = spaces.Box(low=-30, high=30, shape=(1,), dtype=np.int_)
        else:
            self.action_space = spaces.Discrete(len(Action))

    def reset(self):
        self.state.pos = (-1, -1)
        self.state.dir = -1
        self.state.terminated = False
        self.state.carrying = None

    color = PropertyAlias("state", "color", doc="Alias for :attr:`AgentState.color`.")
    dir = PropertyAlias("state", "dir", doc="Alias for :attr:`AgentState.dir`.")
    pos = PropertyAlias("state", "pos", doc="Alias for :attr:`AgentState.pos`.")
    terminated = PropertyAlias(
        "state", "terminated", doc="Alias for :attr:`AgentState.terminated`."
    )
    carrying = PropertyAlias(
        "state", "carrying", doc="Alias for :attr:`AgentState.carrying`."
    )

    @property
    def front_pos(self) -> Position:
        agent_dir = self.state._view[AgentState.DIR]
        agent_pos = self.state._view[AgentState.POS]
        fwd_pos = front_pos(*agent_pos, agent_dir)
        return Position(*fwd_pos)

    def render(self, img: ndarray[np.uint8]):
        """
        Render the agent on the image.
        """
        tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))

        # Rotate agent based on direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=math.radians(self.state.dir))
        fill_coords(img, tri_fn, self.state.color.rgb())


class AgentState(np.ndarray):
    TYPE = 0
    COLOR = 1
    DIR = 2
    POS = slice(3, 5)
    ENCODING = slice(0, 5)
    TERMINATED = 5

    dim = 6
    encode_dim = ENCODING.stop - ENCODING.start + 1  # distance

    def __new__(cls, *dims: int):
        obj = np.zeros(dims + (cls.dim,), dtype=np.int_).view(cls)

        # Set default values
        obj[..., AgentState.TYPE] = WorldObjectType.agent
        obj[..., AgentState.COLOR].flat = Color.cycle(np.prod(dims))
        obj[..., AgentState.DIR] = -1
        obj[..., AgentState.POS] = (-1, -1)

        # Other attributes
        obj._terminated = np.zeros(dims, dtype=bool)  # Cache for faster access
        obj._view = obj.view(
            np.ndarray
        )  # View of the underlying array (faster indexing)
        return obj

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if out.shape and out.shape[-1] == self.dim:
            if not hasattr(self, "_view"):
                logging.warning("AgentState object was not initialized properly.")
                return out
            out._view = self._view[idx, ...]
            out._terminated = self._terminated[idx, ...]
        return out

    @property
    def color(self) -> Color | ndarray[np.str_]:
        """
        Return the agent color.
        """
        return Color.from_index(self._view[..., AgentState.COLOR])

    @color.setter
    def color(self, value: str):
        """
        Set the agent color.
        """
        self[..., AgentState.COLOR] = np.vectorize(lambda c: Color(c).to_index())(value)

    @property
    def dir(self) -> int | ndarray[np.int_]:
        out = self._view[..., AgentState.DIR]
        return out.item() if out.ndim == 0 else out

    @dir.setter
    def dir(self, value: int | ndarray[np.int_]):
        self._view[..., AgentState.DIR] = value

    @property
    def pos(self) -> Position:
        """
        Return the agent's (x, y) position.
        """
        out = self._view[..., AgentState.POS]
        if out.ndim == 1:
            return Position(*out)
        pos = Position.from_list(out)
        return (pos) if pos.ndim == 1 else pos

    @pos.setter
    def pos(self, value: Position | list[int]):
        """
        Set the agent's (x, y) position.
        """
        if isinstance(value, Position):
            value = list(value())
        self[..., AgentState.POS] = value

    @property
    def terminated(self) -> bool | ndarray[np.bool]:
        out = self._terminated
        return out.item() if out.ndim == 0 else out

    @terminated.setter
    def terminated(self, value: bool | ndarray[np.bool]):
        self[..., AgentState.TERMINATED] = value
        self._terminated[...] = value
