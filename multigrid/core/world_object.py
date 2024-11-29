import numpy as np
from multigrid.core.constants import Color, Type
import enum
from numpy.typing import NDArray as ndarray


class WorldObject(np.ndarray):
    TYPE = 0
    COLOR = 1
    STATE = 2

    dim = len([TYPE, COLOR, STATE])

    def __new__(
        cls,
        type_name: str | None = None,
        color: enum.Enum | ndarray = Color.from_index(0),
    ):
        type_idx = Type(type_name).to_index()

        obj = np.zeros(cls.dim, dtype=int).view(cls)
        obj[WorldObject.TYPE] = type_idx
        obj[WorldObject.COLOR] = Color(color).to_index()
        obj._contains: WorldObject | None = None
        obj._init_pos: tuple[int, int] | None = None
        obj._cur_pos: tuple[int, int] | None = None

        return obj

    @staticmethod
    def empty():
        return np.zeros(WorldObject.dim, dtype=int)

    def encode(self) -> tuple[int, int, int]:
        return tuple(self)
