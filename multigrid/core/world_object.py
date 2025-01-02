import functools
import logging
from typing import TYPE_CHECKING, Union, Optional

import numpy as np
from numpy.typing import NDArray

from multigrid.core.constants import Color, State, WorldObjectType
from multigrid.utils.position import Position
from multigrid.utils.rendering import fill_coords, point_in_rect

if TYPE_CHECKING:
    from multigrid.base import MultiGridEnv
    from multigrid.core.agent import Agent


class WorldObjectMeta(type):
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

        if name != "WorldObject":
            type_name = class_dict.get("type_name", name.lower())

            # Add the object class name to the `Type` enumeration if not already present
            if type_name not in set(WorldObjectType):
                WorldObjectType.add_item(type_name, type_name)

            # Store the object class with its corresponding type index
            type_idx = WorldObjectType(type_name).to_index()

            meta._TYPE_IDX_TO_CLASS[type_idx] = cls
        return cls


class WorldObject(np.ndarray, metaclass=WorldObjectMeta):
    TYPE = 0
    COLOR = 1
    STATE = 2

    dim = len([TYPE, COLOR, STATE])

    def __new__(
        cls,
        type_name: str | None = None,
        color: str = Color.from_index(0),
    ) -> "WorldObject":
        type_name = type_name or getattr(cls, "type_name", cls.__name__.lower())
        type_idx = WorldObjectType(type_name).to_index()

        obj = np.zeros(cls.dim, dtype=int).view(cls)
        obj[WorldObject.TYPE] = type_idx
        obj[WorldObject.COLOR] = Color(color).to_index()
        obj._contains: WorldObject | None = None
        obj._init_pos: tuple[int, int] | None = None
        obj._cur_pos: tuple[int, int] | None = None

        return obj

    def __str__(self):
        return f"{self.type}({self.color}, {self.state})"

    @staticmethod
    @functools.cache
    def empty():
        return WorldObject(type_name=WorldObjectType.empty)

    @staticmethod
    def from_array(arr: list[int]) -> Optional["WorldObject"]:
        type_idx = arr[WorldObject.TYPE]

        if type_idx == WorldObjectType.empty.to_index():
            return None

        if type_idx in WorldObject._TYPE_IDX_TO_CLASS:
            cls = WorldObject._TYPE_IDX_TO_CLASS[type_idx]
            obj = cls.__new__(cls)
            obj[...] = arr
            return obj

        raise ValueError(f"Unknown object type index: {type_idx}")

    @functools.cached_property
    def type(self) -> WorldObjectType:
        """
        Return the object type.
        """
        return WorldObjectType.from_index(self[WorldObject.TYPE])

    @property
    def color(self) -> Color:
        """
        Return the object color.
        """
        return Color.from_index(self[WorldObject.COLOR])

    @color.setter
    def color(self, value: str):
        """
        Set the object color.
        """
        self[WorldObject.COLOR] = Color(value).to_index()

    @property
    def state(self) -> str:
        """
        Return the name of the object state.
        """
        return State.from_index(self[WorldObject.STATE])

    @state.setter
    def state(self, value: str):
        """
        Set the name of the object state.
        """
        self[WorldObject.STATE] = State(value).to_index()

    def can_overlap(self) -> bool:
        """
        Can an agent overlap with this?
        """
        return self.type == WorldObjectType.empty

    def can_pickup(self) -> bool:
        """
        Can an agent pick this up?
        """
        return False

    def can_contain(self) -> bool:
        """
        Can this contain another object?
        """
        return False

    def can_place(self) -> bool:
        """
        Can this object be placed on top of another object?
        """
        return False

    def toggle(self, env: "MultiGridEnv", agent: "Agent", pos: Position) -> bool:
        """
        Toggle the state of this object or trigger an action this object performs.

        Parameters
        ----------
        env : MultiGridEnv
            The environment this object is contained in
        agent : Agent
            The agent performing the toggle action
        pos : tuple[int, int]
            The (x, y) position of this object in the environment grid

        Returns
        -------
        success : bool
            Whether the toggle action was successful
        """
        return False

    def encode(self) -> tuple[int, int, int]:
        """
        Encode a 3-tuple description of this object.

        Returns
        -------
        type_idx : int
            The index of the object type
        color_idx : int
            The index of the object color
        state_idx : int
            The index of the object state
        """
        return tuple(self)

    @staticmethod
    def decode(
        type_idx: int, color_idx: int, state_idx: int
    ) -> Union["WorldObject", None]:
        """
        Create an object from a 3-tuple description.

        Parameters
        ----------
        type_idx : int
            The index of the object type
        color_idx : int
            The index of the object color
        state_idx : int
            The index of the object state
        """
        arr = np.array([type_idx, color_idx, state_idx])
        return WorldObject.from_array(arr)

    def render(self, img: NDArray[np.uint8]):
        raise NotImplementedError


class Goal(WorldObject):
    """
    Goal object an agent may be searching for.
    """

    def __new__(cls, color: str = Color.green):
        return super().__new__(cls, color=color)

    def can_overlap(self) -> bool:
        """
        :meta private:
        """
        return True

    def render(self, img):
        """
        :meta private:
        """
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())


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

    def can_place(self) -> bool:
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

    def can_overlap(self) -> bool:
        """
        :meta private:
        """
        return False

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

    def toggle(self, env, agent, pos: Position):
        """
        :meta private:
        """
        # Replace the box by its contents
        env.grid.set(pos, self.contains)
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


class Container(WorldObject):
    """
    Container object that may contain one object.
    """

    def __new__(cls, color: str = Color.purple, contains: WorldObject | None = None):
        """
        Parameters
        ----------
        color : str
            Object color
        contains : WorldObj or None
            Object contents
        """
        container = super().__new__(cls, color=color)
        container.contains = contains
        return container

    def can_pickup(self) -> bool:
        """
        :meta private:
        """
        return False

    def can_overlap(self) -> bool:
        """
        :meta private:
        """
        if self.contains is not None:
            return self.contains.can_overlap()
        return True

    def can_contain(self) -> bool:
        """
        :meta private:
        """
        if self.contains is not None:
            return False
        return True

    def can_pickup_contained(self) -> bool:
        """
        : meta private:
        """
        return False

    def render(self, img):
        """
        :meta private:
        """
        color = self.color.rgb() / 5
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)
        if self.contains is not None:
            self.contains.render(img)

    @property
    def contains(self) -> WorldObject | None:
        """
        Get the object contained in this container.
        """
        assert (self.state == State.contained) == (
            self._contains is not None
        ), f"State and object consistency mismatch: {self.state}, {self._contains}"
        return self._contains

    @contains.setter
    def contains(self, obj: WorldObject | None):
        """
        Set the object contained in this container.
        """
        self._contains = obj

        if obj is None:
            self.state = State.empty
        else:
            self.state = State.contained
