import functools
from ..core.constants import Direction


@functools.cache
def front_pos(agent_x: int, agent_y: int, agent_dir: int):
    """
    Get the position in front of an agent.
    """
    dx, dy = Direction(agent_dir).to_vec()
    return (agent_x + dx, agent_y + dy)
