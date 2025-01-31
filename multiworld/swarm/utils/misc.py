import functools
import math

from typing import Tuple


def are_within_radius(tuple0: Tuple[int, int], tuple1: Tuple[int, int], radius: float):
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(tuple0, tuple1)))
    return distance <= radius


@functools.cache
def front_pos(agent_x: int, agent_y: int, agent_dir: int):
    """
    Get the position in front of an agent.
    """
    direction_radians = math.radians(agent_dir)

    delta_x = math.cos(direction_radians)
    delta_y = math.sin(direction_radians)

    new_x = agent_x + round(delta_x)
    new_y = agent_y + round(delta_y)

    return new_x, new_y
