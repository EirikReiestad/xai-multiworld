import enum
import functools
from typing import Union

import numpy as np
from numpy.typing import NDArray

from multigrid.core.constants import Color, Type
from multigrid.utils.rendering import fill_coords, point_in_rect


class WorldObjMeta(type):
    """
    Metaclass for world objects.

    Each subclass is associated with a unique :class:`Type` enumeration value.

    By default, the type name is the class name (in lowercase), but this can be
    overridden by setting the `type_name` attribute in the class definition.
    Type names are dynamically added to the :class:`Type` enumeration
    if not already present.

    Examples
    --------
    >>> class A(WorldObj): pass
    >>> A().type
    <Type.a: 'a'>

    >>> class B(WorldObj): type_name = 'goal'
    >>> B().type
    <Type.goal: 'goal'>

    :meta private:
    """

    # Registry of object classes
    _TYPE_IDX_TO_CLASS = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        if name != "WorldObj":
            type_name = class_dict.get("type_name", name.lower())

            # Add the object class name to the `Type` enumeration if not already present
            if type_name not in set(Type):
                Type.add_item(type_name, type_name)

            # Store the object class with its corresponding type index
            meta._TYPE_IDX_TO_CLASS[Type(type_name).to_index()] = cls

        return cls


class WorldObject(np.ndarray, metaclass=WorldObjMeta):
    TYPE = 0
    COLOR = 1
    STATE = 2

    dim = len([TYPE, COLOR, STATE])

    def __new__(
        cls,
        type_name: str | None = None,
        color: enum.Enum | NDArray = Color.from_index(0),
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

    @staticmethod
    def from_array(arr: list[int]) -> Union["WorldObject", None]:
        type_idx = arr[WorldObject.TYPE]

        if type_idx == Type.empty.to_index():
            return None

        if type_idx in WorldObject._TYPE_IDX_TO_CLASS:
            cls = WorldObject._TYPE_IDX_TO_CLASS[type_idx]
            obj = cls.__new__(cls)
            obj[...] = arr
            return obj

    def encode(self) -> tuple[int, int, int]:
        return tuple(self)

    def render(self, img: NDArray[np.uint8]):
        raise NotImplementedError


class Floor(WorldObject):
    """
    Colored floor tile an agent can walk over.
    """

    def __new__(cls, color: str = Color.blue):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, color=color)

    def can_overlap(self) -> bool:
        """
        :meta private:
        """
        return True

    def render(self, img: NDArray[np.uint8]):
        """
        :meta private:
        """
        # Give the floor a pale color
        color = self.color.rgb() / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Wall(WorldObject):
    """
    Wall object that agents cannot move through.
    """

    @functools.cache  # reuse instances, since object is effectively immutable
    def __new__(cls, color: str = Color.grey):
        """
        Parameters
        ----------
        color : str
            Object color
        """
        return super().__new__(cls, color=color)

    def render(self, img):
        """
        :meta private:
        """
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())


class Box(WorldObject):
    """
    Box object that may contain other objects.
    """

    def __new__(cls, color: str = Color.yellow, contains: WorldObject | None = None):
        """
        Parameters
        ----------
        color : str
            Object color
        contains : WorldObj or None
            Object contents
        """
        box = super().__new__(cls, color=color)
        box.contains = contains
        return box

    def can_pickup(self) -> bool:
        """
        :meta private:
        """
        return True

    def can_contain(self) -> bool:
        """
        :meta private:
        """
        return False

    def toggle(self, env, agent, pos):
        """
        :meta private:
        """
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

    def render(self, img):
        """
        :meta private:
        """
        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), self.color.rgb())
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), self.color.rgb())
