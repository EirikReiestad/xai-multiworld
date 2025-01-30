from multiworld.swarm.base import SwarmEnv
from multiworld.swarm.core.world import World


class EmptyEnv(SwarmEnv):
    def _gen_world(self, width: int, height: int):
        self.world = World(width, height, self._object_size)

        for agent in self.agents:
            position = self.world.get_empty_position(self.np_random)
            agent.state.pos = position
