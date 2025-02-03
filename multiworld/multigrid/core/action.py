import enum
from typing import Dict

from multiworld.utils.typing import AgentID


class Action(enum.IntEnum):
    """
    Enumeration of possible actions.
    """

    left = 0  #: Turn left
    right = enum.auto()  #: Turn right
    forward = enum.auto()  #: Move forward
    # pickup = enum.auto()  #: Pick up an object
    # drop = enum.auto()  #: Drop an object
    # toggle = enum.auto()  #: Toggle / activate an object
    # done = enum.auto()  #: Done completing task


def int_to_action(actions: Dict[AgentID, Action | int]) -> Dict[AgentID, Action]:
    """
    Convert integer action to Action.
    """
    return {agent_id: Action(action) for agent_id, action in actions.items()}
