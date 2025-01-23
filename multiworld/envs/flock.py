from multiworld.base import MultiWorldEnv
from multiworld.core.world import World


class FlockEnv(MultiWorldEnv):
    def _gen_world(self, width: int, height: int):
        self.world = World(width, height, self._object_size)

        for agent in self.agents:
            position = self.world.get_empty_position(self.np_random)
            agent.state.pos = position
