from multiworld.base import MultiWorldEnv
from multiworld.core.world import World
from multiworld.core.world_object import Box, Container
from multiworld.core.area import Area
from multiworld.utils.position import Position


class EmptyEnv(MultiWorldEnv):
    def _gen_world(self, width: int, height: int):
        self.world = World(width, height)

        for agent in self.agents:
            position = self.world.get_empty_position(self.np_random)
            agent.state.pos = position
