import enum
from typing import Dict

from swarm.utils.typing import AgentID


class Action(enum.IntEnum):
    """
    Enumeration of possible actions.
    """

    left90 = 0  #: Turn left
    left45 = enum.auto()
    forward = enum.auto()
    right45 = enum.auto()  #: Turn right
    right90 = enum.auto()  #: Turn right


def int_to_action(actions: Dict[AgentID, Action | int]) -> Dict[AgentID, Action]:
    """
    Convert integer action to Action.
    """
    return {agent_id: Action(action) for agent_id, action in actions.items()}
