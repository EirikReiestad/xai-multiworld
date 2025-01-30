from multiworld.multigrid.core.action import Action as MultiGridAction
from multiworld.multigrid.core.agent import (
    Agent as MultiGridAgent,
)
from multiworld.multigrid.core.agent import (
    AgentState as MultiGridAgentState,
)
from multiworld.multigrid.core.constants import Direction
from multiworld.multigrid.core.world_object import (
    WorldObject as MultiGridWorldObject,
)
from multiworld.swarm.core.action import Action as SwarmAction
from multiworld.swarm.core.agent import (
    Agent as SwarmAgent,
)
from multiworld.swarm.core.agent import (
    AgentState as SwarmAgentState,
)
from multiworld.swarm.core.world_object import WorldObject as SwarmWorldObject

WorldObject = SwarmWorldObject | MultiGridWorldObject

Agent = SwarmAgent | MultiGridAgent

AgentState = SwarmAgentState | MultiGridAgentState

Action = SwarmAction | MultiGridAction

Direction = Direction
