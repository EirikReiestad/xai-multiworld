from typing import List

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray as ndarray

from swarm.core.agent import Agent, AgentState
from swarm.core.constants import Color, WorldObjectType
from swarm.core.world_object import Wall, WorldObject
from swarm.utils.position import Position

WALL_ENCODING = Wall().encode()
UNSEEN_ENCODING = WorldObject(WorldObjectType.unseen, Color.from_index(0)).encode()
EMPTY_ENCODING = WorldObject(WorldObjectType.empty, Color.from_index(0)).encode()
ENCODE_DIM = WorldObject.dim

GRID_ENCODING_IDX = slice(None)

AGENT_DIR_IDX = AgentState.DIR
AGENT_POS_IDX = AgentState.POS
AGENT_TERMINATED_IDX = AgentState.TERMINATED
AGENT_ENCODING_IDX = AgentState.ENCODING
AGENT_ENCODE_DIM = AgentState.encode_dim

TYPE = WorldObject.TYPE
STATE = WorldObject.STATE

WALL = int(WorldObjectType.wall)
BOX = int(WorldObjectType.box)


def gen_obs_grid_encoding(
    agent_state: np.ndarray,
    agent_view_size: int,
    max_observations: int,
) -> np.ndarray:
    """
    This function returns the encoded agents that are within a certain radius of the observing agent.
    The closest agents will be prioritized and only the first `max_observations` agents will be included.
    """

    agent_grid = agent_state[..., AGENT_ENCODING_IDX]
    agent_pos = agent_state[..., AGENT_POS_IDX]
    agent_terminated = agent_state[..., AGENT_TERMINATED_IDX]

    num_agents = len(agent_state)
    obs = np.zeros(
        (num_agents, 1, max_observations, AGENT_ENCODE_DIM), dtype=np.float32
    )

    for agent in range(num_agents):
        if agent_terminated[agent]:
            continue

        pos = agent_pos[agent]
        distances = np.linalg.norm(agent_pos - pos, axis=1)

        valid_mask = (distances <= agent_view_size) & (~agent_terminated)
        valid_agents = np.where(valid_mask)[0]
        valid_distances = distances[valid_agents]
        valid_agent_grid = agent_grid[valid_agents]

        sorted_indices = np.argsort(valid_distances)
        sorted_valid_agents = valid_agents[sorted_indices]
        sorted_valid_distances = valid_distances[sorted_indices]
        sorted_valid_agent_grid = valid_agent_grid[sorted_indices]

        num_observations = min(len(sorted_valid_agents), max_observations)
        masked_grid = np.zeros((max_observations, AGENT_ENCODE_DIM), dtype=np.float32)

        for i in range(num_observations):
            masked_grid[i, : agent_grid.shape[1]] = sorted_valid_agent_grid[i]
            normalized_distance = sorted_valid_distances[i] / agent_view_size
            masked_grid[i, -1] = normalized_distance

        obs[agent] = [masked_grid]

    return obs


def see_behind(world_object: ndarray[np.int_] | None) -> bool:
    """
    Can an agent see behind this object?

    Parameters
    ----------
    world_obj : ndarray[int] of shape (encode_dim,)
        World object encoding
    """
    if world_object is None:
        return True
    if world_object[TYPE] == WALL:
        return False

    return True


def get_see_behind_mask(grid_array: ndarray[np.int_]) -> ndarray[np.bool_]:
    """
    Return boolean mask indicating which grid locations can be seen through.

    Parameters
    ----------
    grid_array : ndarray[int] of shape (num_agents, width, height, dim)
        Grid object array for each agent

    Returns
    -------
    see_behind_mask : ndarray[bool] of shape (width, height)
        Boolean visibility mask
    """
    num_agents, height, width = grid_array.shape[:3]
    see_behind_mask = np.zeros((num_agents, height, width), dtype=np.bool_)
    for agent in range(num_agents):
        for i in range(height):
            for j in range(width):
                see_behind_mask[agent, i, j] = see_behind(grid_array[agent, i, j])

    return see_behind_mask


def get_vis_mask(obs_grid: ndarray[np.int_]) -> ndarray[np.bool_]:
    """
    Generate a boolean mask indicating which grid locations are visible to each agent.

    Parameters
    ----------
    obs_grid : ndarray[int] of shape (num_agents, width, height, dim)
        Grid object array for each agent observation

    Returns
    -------
    vis_mask : ndarray[bool] of shape (num_agents, width, height)
        Boolean visibility mask for each agent
    """
    num_agents, height, width = obs_grid.shape[:3]
    see_behind_mask = get_see_behind_mask(obs_grid)
    vis_mask = np.zeros((num_agents, width, height), dtype=np.bool_)
    vis_mask[:, height - 1, width // 2] = True  # agent relative position

    for agent in range(num_agents):
        for j in range(height - 1, -1, -1):
            # Right propegate
            for i in range(width // 2, width):
                if not vis_mask[agent, j, i] or not see_behind_mask[agent, j, i]:
                    continue
                vis_mask[agent, j - 1, i] = True
                if i + 1 < width:
                    vis_mask[agent, j - 1, i + 1] = True
                    vis_mask[agent, j, i + 1] = True
            # Left propegate
            for i in range(width // 2, -1, -1):
                if not vis_mask[agent, j, i] or not see_behind_mask[agent, j, i]:
                    continue
                vis_mask[agent, j - 1, i] = True
                if i - 1 >= 0:
                    vis_mask[agent, j - 1, i - 1] = True
                    vis_mask[agent, j, i - 1] = True

    return vis_mask


def agents_from_agent_observation(obs_grid: ndarray[np.int_]) -> List[Agent]:
    width, height = obs_grid.shape[0], obs_grid.shape[1]
    agent_encodings = []
    agents_pos = []
    for y in range(height):
        for x in range(width):
            pos = Position(x, y)
            cell = obs_grid[pos.y, pos.x]
            if cell[TYPE] == WorldObjectType.agent.to_index():
                agent_encodings.append(cell.copy())
                agents_pos.append(pos())
                obs_grid[pos.y, pos.x] = EMPTY_ENCODING
            # TODO: This should be handled differently
            if cell[TYPE] == WorldObjectType.unseen.to_index():
                obs_grid[pos.y, pos.x] = EMPTY_ENCODING

    agent_state = AgentState(len(agent_encodings))

    for i in range(len(agent_encodings)):
        agent_state[i, AGENT_ENCODING_IDX] = agent_encodings[i]
        agent_state[i, AGENT_POS_IDX] = agents_pos[i]

    agents = []
    for i in range(len(agent_encodings)):
        agent = Agent(i)
        agent.state = agent_state[agent.index]
        agents.append(agent)

    return agents
