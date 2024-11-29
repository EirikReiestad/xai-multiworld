from multigrid.core.world_object import WorldObject
import numpy as np
from numpy.typing import NDArray as ndarray
from gymnasium import spaces
from multigrid.core.action import Action
from multigrid.core.constants import Direction, Type, Color
from multigrid.utils.property_alias import PropertyAlias


class Agent:
    def __init__(self, index: int, view_size: int = 7, see_through_walls: bool = False):
        self.index = index
        assert view_size % 2 == 1, "View size must be odd for agent observation."
        assert view_size > 1, "View size must be greater than 1 for agent observation."
        self.view_size = view_size
        self.state = AgentState()

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(view_size, view_size, WorldObject.dim),
                    dtype=np.int_,
                ),
                "direction": spaces.Discrete(len(Direction)),
            }
        )
        self.action_space = spaces.Discrete(len(Action))

    def reset(self):
        pass

    def action_space(self):
        pass

    dir = PropertyAlias("state", "dir", doc="Alias for :attr:`AgentState.dir`.")


class AgentState(np.ndarray):
    TYPE = 0
    COLOR = 1
    DIR = 2
    ENCODING = slice(0, 3)
    POS = slice(3, 5)
    TERMINATED = 5
    CARRYING = slice(6, 6 + WorldObject.dim)

    dim = 6 + WorldObject.dim

    def __new__(cls, *dims: int):
        obj = np.zeros(dims + (cls.dim,), dtype=np.int_).view(cls)

        # Set default values
        obj[..., AgentState.TYPE] = Type.agent
        obj[..., AgentState.COLOR].flat = Color.cycle(np.prod(dims))
        obj[..., AgentState.DIR] = -1
        obj[..., AgentState.POS] = (-1, -1)

        # Other attributes
        obj._carried_obj = np.zeros(dims, dtype=object)  # Object references
        obj._terminated = np.zeros(dims, dtype=bool)  # Cache for faster access
        obj._view = obj.view(
            np.ndarray
        )  # View of the underlying array (faster indexing)
        return obj

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if out.shape and out.shape[-1] == self.dim:
            out._view = self._view[idx, ...]
            out._carried_obj = self._carried_obj[idx, ...]
            out._terminated = self._terminated[idx, ...]
        return out

    def reset(self):
        pass

    @property
    def dir(self) -> Direction | ndarray[np.int_]:
        out = self._view[..., AgentState.DIR]
        return Direction(out.item()) if out.ndim == 0 else out

    @dir.setter
    def dir(self, value: Direction | ndarray[np.int_]):
        self._view[..., AgentState.DIR] = value

    @property
    def terminated(self) -> bool | ndarray[np.bool]:
        out = self._terminated
        return out.item() if out.ndim == 0 else out

    @terminated.setter
    def terminated(self, value: bool | ndarray[np.bool]):
        self[..., AgentState.TERMINATED] = value
        self._terminated[...] = value
