from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from multiworld.core.position import Position
from multiworld.swarm.core.constants import OBJECT_SIZE
from multiworld.swarm.core.world_object import WorldObject
from multiworld.utils.random import RandomMixin
from multiworld.utils.rendering import (
    downsample,
    highlight_img,
)

from .agent import Agent


class World:
    _object_cache = {}

    def __init__(self, width: int, height: int, object_size: int = OBJECT_SIZE):
        assert width >= 3
        assert height >= 3
        self.width = width
        self.height = height
        self.object_size = object_size

        self.world_objects: Dict[Tuple[int, int], WorldObject] = {}

        self.state: NDArray[np.int_] = np.zeros(
            (width, height, WorldObject.dim), dtype=int
        )
        self.state[...] = WorldObject.empty()

    @classmethod
    def render_object(
        cls,
        obj: WorldObject | None = None,
        agent: Agent | None = None,
        highlight: bool = False,
        object_size: int = OBJECT_SIZE,
        subdivs: int = 3,
    ) -> NDArray[np.uint8]:
        # Hashmap lookup for the cache
        assert obj is None or isinstance(obj, WorldObject)
        key: tuple[Any, ...] = (highlight, object_size)
        if agent:
            key += (agent.state.color, agent.state.dir)
        else:
            key += (None, None)

        obj_encode = (
            obj.encode() if isinstance(obj, np.ndarray) and obj.size > 0 else None
        )

        key = obj_encode + key if obj_encode else key

        if key in cls._object_cache:
            return cls._object_cache[key]

        img = np.zeros(
            shape=(object_size * subdivs, object_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        # fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        # fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        if agent is not None and not agent.state.terminated:
            agent.render(img)

        if highlight:
            highlight_img(img)

        img = downsample(img, subdivs)

        cls._object_cache[key] = img
        return img

    def render(
        self,
        object_size: int,
        agents: Iterable[Agent] = (),
        world_objects: Iterable[WorldObject] = (),
        highlight_mask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.uint8]:
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Get agent locations
        # For overlapping agents, non-terminated agents is prioritized
        location_to_agent: Dict[Tuple[Any, Any], Optional[Agent]] = {}
        for agent in sorted(agents, key=lambda x: not x.terminated):
            location_to_agent[agent.pos()] = agent

        # Initialize pixel array
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for obj in world_objects:
            if obj is None:
                continue
            xmin = obj.pos.x
            xmax = min(obj.pos.x + obj.object_size, self.width)
            ymin = obj.pos.y
            ymax = min(obj.pos.y + obj.object_size, self.height)
            assert (
                ymin >= 0 - object_size
                and ymax <= self.height + object_size
                and ymin < ymax
            )
            assert (
                xmin >= 0 - object_size
                and xmax <= self.width + object_size
                and xmin < xmax
            )
            object_img = World.render_object(obj, object_size=obj.object_size)
            mask = np.all(img[ymin:ymax, xmin:xmax, :].copy() == 0, axis=-1)
            img[ymin:ymax, xmin:xmax, :][mask] = object_img[mask]

        for agent in agents:
            xmin = agent.pos.x
            xmax = min(agent.pos.x + object_size, self.width)
            ymin = agent.pos.y
            ymax = min(agent.pos.y + object_size, self.height)

            assert (
                ymin >= 0 - object_size
                and ymax <= self.height + object_size
                and ymin < ymax
                and ymax - ymin == object_size
            )
            assert (
                xmin >= 0 - object_size
                and xmax <= self.width + object_size
                and xmin < xmax
                and xmax - xmin == object_size
            )
            cell = self.get(agent.pos)
            object_img = World.render_object(
                cell,
                agent=agent,
                highlight=False,
                object_size=object_size,
            )
            mask = np.all(img[ymin:ymax, xmin:xmax, :].copy() == 0, axis=-1)
            img[ymin:ymax, xmin:xmax, :][mask] = object_img[mask]

        return img

    def in_bounds(self, position: Position | list[Position]) -> bool | list[bool]:
        def in_bound(p: Position):
            x = 0 <= p.x < self.width - self.object_size
            y = 0 <= p.y < self.height - self.object_size
            return x and y

        if isinstance(position, Position):
            return in_bound(position)
        return [in_bound(pos) for pos in position]

    def get(self, pos: Position) -> WorldObject | None:
        if not self.in_bounds(pos):
            return None
        if (pos.x, pos.y) not in self.world_objects:
            obj = WorldObject.from_array(self.state[pos.x, pos.y])
            self.world_objects[pos.x, pos.y] = obj
        return self.world_objects[pos.x, pos.y]

    def set(self, pos: Position, obj: WorldObject | None):
        if not self.in_bounds(pos):
            return
        self.world_objects[pos.x, pos.y] = obj
        if isinstance(obj, WorldObject):
            self.state[pos.x, pos.y] = obj
        elif obj is None:
            self.state[pos.x, pos.y] = WorldObject.empty()
        else:
            raise TypeError(f"Cannot set grid value to {type(obj)}")

    def get_empty_position(
        self, random_generator: Generator, max_iter: int = 100
    ) -> Position:
        random = RandomMixin(random_generator)
        if max_iter == 0:
            raise RuntimeError("Could not find a position")
        random_position = random._rand_pos(0, self.width, 0, self.height)
        if self.world_objects.get(random_position()) is None and self.in_bounds(
            random_position
        ):
            return random_position
        return self.get_empty_position(random_generator, max_iter - 1)

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height
