from collections import defaultdict
from typing import Any, Iterable, Optional

from multigrid.core.constants import TILE_PIXELS

import numpy as np
from numpy.typing import NDArray

from multigrid.core.world_object import WorldObject

from multigrid.utils.rendering import downsample, fill_coords, point_in_rect

from .agent import Agent


class Grid:
    _tile_cache = {}

    def __init__(self, width: int, height: int):
        assert width >= 3
        assert height >= 3
        self._width = width
        self._height = height

        self._world_objects: dict[
            tuple[int, int], WorldObject
        ] = {}  # index by position
        self.state: NDArray[np.int_] = np.zeros(
            (width, height, WorldObject.dim), dtype=int
        )
        self.state[...] = WorldObject.empty()

    @classmethod
    def render_tile(
        cls,
        obj: WorldObject | None = None,
        agent: Agent | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
    ) -> NDArray[np.uint8]:
        # Hashmap lookup for the cache
        key: tuple[Any, ...] = (highlight, tile_size)
        if agent:
            key += (agent.state.color, agent.state.dir)
        else:
            key += (None, None)
        key = obj.encode() + key if obj else key

        if key in cls._tile_cache:
            return cls._tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        if agent is not None and not agent.state.terminated:
            agent.render(img)

        if highlight:
            highlight_img(img)

        img = downsample(img, subdivs)

        cls._tile_cache[key] = img
        return img

    def render(
        self,
        tile_size: int,
        agents: Iterable[Agent] = (),
        highlight_mask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.uint8]:
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self._width, self._height), dtype=bool)

        # Get agent locations
        # For overlapping agents, non-terminated agents is prioritized
        location_to_agent: dict[tuple[Any, Any], Optional[Agent]] = {}
        for agent in sorted(agents, key=lambda x: not x.terminated):
            location_to_agent[tuple(agent.pos)] = agent

        # Initialize pixel array
        width_px = self._width * tile_size
        height_px = self._height * tile_size
        img = np.zeros((height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self._height):
            for i in range(0, self._width):
                assert highlight_mask is not None
                cell = self.get(i, j)
                tile_img = Grid.render_tile(
                    cell,
                    agent=location_to_agent.get((i, j)),
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )
                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                assert ymin >= 0 and ymax <= height_px and ymin < ymax
                assert xmin >= 0 and xmax <= width_px and xmin < xmax
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def get(self, x: int, y: int) -> WorldObject | None:
        if (x, y) not in self._world_objects:
            self._world_objects[x, y] = WorldObject.from_array(self.state[x, y])
        return self._world_objects[x, y]

    def set(self, x: int, y: int, obj: WorldObject | None):
        if isinstance(obj, WorldObject):
            self.state[x, y] = obj
        elif obj is None:
            self.state[x, y] = WorldObject.empty()
        else:
            raise TypeError(f"Cannot set grid value to {type(obj)}")
