from typing import List

import numpy as np
from numpy.typing import NDArray as ndarray

from utils.core.constants import Color
from multiworld.core.position import Position
from multiworld.multigrid.core.agent import Agent, AgentState
from multiworld.multigrid.core.constants import Direction, State, WorldObjectType
from multiworld.multigrid.core.world_object import Wall, WorldObject
from multiworld.multigrid.utils.ohe import (
    OHE_GRID_OBJECT_DIM,
    OHE_GRID_OBJECT_DIM_MINIMAL,
    ohe_grid_object,
)
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum

WALL_ENCODING = Wall().encode()
UNSEEN_ENCODING = WorldObject(WorldObjectType.unseen, Color.from_index(0)).encode()
EMPTY_ENCODING = WorldObject(WorldObjectType.empty, Color.from_index(0)).encode()
ENCODE_DIM = WorldObject.dim

GRID_ENCODING_IDX = slice(None)

AGENT_DIR_IDX = AgentState.DIR
AGENT_POS_IDX = AgentState.POS
AGENT_TERMINATED_IDX = AgentState.TERMINATED
AGENT_CARRYING_IDX = AgentState.CARRYING
AGENT_ENCODING_IDX = AgentState.ENCODING

TYPE = WorldObject.TYPE
STATE = WorldObject.STATE

WALL = int(WorldObjectType.wall)
BOX = int(WorldObjectType.box)

RIGHT = int(Direction.right)
LEFT = int(Direction.left)
UP = int(Direction.up)
DOWN = int(Direction.down)


def pertubate_observation(
    grid_state: ndarray[np.int_],
    idx: List[int],
) -> List[ndarray[np.int_]]:
    observations = []

    types = [
        WorldObjectType.empty,
        WorldObjectType.goal,
        WorldObjectType.agent,
        WorldObjectType.wall,
    ]
    colors = [Color.red, Color.green, Color.blue, Color.purple, Color.yellow]
    states = [State.empty]

    if idx[3] == 0:
        for t in types:
            state = grid_state.copy()
            state[idx[0]][idx[1]][idx[2]][idx[3]] = int(t)
            observations.append(state)
        return observations
    if idx[3] == 1:
        for c in colors:
            state = grid_state.copy()
            state[idx[0]][idx[1]][idx[2]][idx[3]] = int(c)
            observations.append(state)
        return observations
    if idx[3] == 2:
        for s in states:
            state = grid_state.copy()
            state[idx[0]][idx[1]][idx[2]][idx[3]] = int(s)
            observations.append(state)
        return observations

    return observations


def gen_obs_grid_encoding(
    grid_state: ndarray[np.int_],
    agent_state: ndarray[np.int_],
    agent_view_size: int | None,
    see_through_walls: bool,
    preprocessing: PreprocessingEnum,
) -> ndarray[np.int_]:
    num_agents = len(agent_state)
    obs_grid = gen_obs_grid(grid_state, agent_state, agent_view_size, preprocessing)
    if agent_view_size is None:
        return obs_grid
    # Generate and apply visability mask
    vis_mask = get_vis_mask(obs_grid)
    if see_through_walls:
        return obs_grid
    for agent in range(num_agents):
        for i in range(agent_view_size):
            for j in range(agent_view_size):
                if not vis_mask[agent, i, j]:
                    obs_grid[agent, i, j] = UNSEEN_ENCODING
    return obs_grid


def gen_obs_grid(
    grid_state: ndarray[np.int_],
    agent_state: ndarray[np.int_],
    agent_view_size: int | None,
    preprocessing: PreprocessingEnum,
) -> ndarray[np.int_]:
    num_agents = len(agent_state)

    # Process agent states
    agent_grid = agent_state[..., AGENT_ENCODING_IDX]
    agent_dir = agent_state[..., AGENT_DIR_IDX]
    agent_pos = agent_state[..., AGENT_POS_IDX]
    agent_terminated = agent_state[..., AGENT_TERMINATED_IDX]
    agent_carrying = agent_state[..., AGENT_CARRYING_IDX]

    if num_agents > 0:
        grid_encoding = np.empty((*grid_state.shape[:-1], ENCODE_DIM), dtype=np.int_)
        grid_encoding[...] = grid_state[..., GRID_ENCODING_IDX]

        # Insert agent grid encodings
        for agent in range(num_agents):
            if agent_terminated[agent]:
                continue
            x, y = agent_pos[agent]
            grid_encoding[x, y, GRID_ENCODING_IDX] = agent_grid[agent]
    else:
        grid_encoding = grid_state[..., GRID_ENCODING_IDX]

    ohe_minimal = preprocessing == PreprocessingEnum.ohe_minimal
    ohe = preprocessing == PreprocessingEnum.ohe
    ohe_grid_encoding_dim = None

    if ohe:
        ohe_grid_encoding_dim = OHE_GRID_OBJECT_DIM
    elif ohe_minimal:
        ohe_grid_encoding_dim = OHE_GRID_OBJECT_DIM_MINIMAL

    if ohe_grid_encoding_dim is not None:
        ohe_grid_encoding = np.empty(
            (*grid_state.shape[:-1], ohe_grid_encoding_dim), dtype=np.int_
        )
        for y in range(grid_encoding.shape[1]):
            for x in range(grid_encoding.shape[0]):
                ohe_grid_obj = ohe_grid_object(grid_encoding[x, y], ohe_minimal)
                ohe_grid_encoding[x, y] = ohe_grid_obj
        grid_encoding = ohe_grid_encoding

    if agent_view_size is None:
        width = grid_state.shape[0]
        height = grid_state.shape[1]
        obs_grid = np.empty((num_agents, height, width, ENCODE_DIM), dtype=np.int_)
        for agent in range(num_agents):
            obs_grid[agent, ...] = grid_encoding.copy()
        return obs_grid

    obs_width, obs_height = agent_view_size, agent_view_size

    top_left = get_view_exts(agent_dir, agent_pos, agent_view_size)
    topX, topY = top_left[:, 0], top_left[:, 1]

    num_left_rotations = (agent_dir + 1) % 4
    if ohe:
        obs_grid = np.empty(
            (num_agents, obs_height, obs_width, OHE_GRID_OBJECT_DIM), dtype=np.int_
        )
    elif ohe_minimal:
        obs_grid = np.empty(
            (num_agents, obs_height, obs_width, OHE_GRID_OBJECT_DIM_MINIMAL),
            dtype=np.int_,
        )
    else:
        obs_grid = np.empty(
            (num_agents, obs_height, obs_width, ENCODE_DIM), dtype=np.int_
        )

    for agent in range(num_agents):
        for j in range(obs_height):
            for i in range(obs_width):
                x, y = topX[agent] + i, topY[agent] + j
                # Rotated relative coordinates for observation grid
                if num_left_rotations[agent] == 0:
                    i_rot, j_rot = i, j
                elif num_left_rotations[agent] == 1:
                    i_rot, j_rot = j, obs_width - 1 - i
                elif num_left_rotations[agent] == 2:
                    i_rot, j_rot = obs_width - 1 - i, obs_height - 1 - j
                elif num_left_rotations[agent] == 3:
                    i_rot, j_rot = obs_height - 1 - j, i
                else:
                    raise ValueError("Invalid rotation")

                # Set observation grid
                if 0 <= x < grid_state.shape[0] and 0 <= y < grid_state.shape[1]:
                    obs_grid[agent, j_rot, i_rot] = grid_encoding[x, y]
                else:
                    if ohe or ohe_minimal:
                        obs_grid[agent, j_rot, i_rot] = ohe_grid_object(
                            np.array(WALL_ENCODING), ohe_minimal
                        )
                    else:
                        obs_grid[agent, j_rot, i_rot] = WALL_ENCODING

    # Make it so the agent sees what it is carrying
    if ohe_grid_encoding_dim is not None:
        ohe_agent_carrying = np.zeros(
            (num_agents, ohe_grid_encoding_dim),
            dtype=np.int_,
        )
        for agent in range(num_agents):
            carrying = ohe_grid_object(agent_carrying[agent], ohe_minimal)
            ohe_agent_carrying[agent] = carrying
        agent_carrying = ohe_agent_carrying
    obs_grid[:, obs_height - 1, obs_width // 2] = agent_carrying
    return obs_grid


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


def get_view_exts(
    agent_dir: ndarray[np.int_], agent_pos: ndarray[np.int_], agent_view_size: int
) -> ndarray[np.int_]:
    agent_x, agent_y = agent_pos[:, 0], agent_pos[:, 1]
    top_left = np.zeros((len(agent_dir), 2), dtype=np.int_)

    # Facing right
    top_left[agent_dir == RIGHT, 0] = agent_x[agent_dir == RIGHT]
    top_left[agent_dir == RIGHT, 1] = agent_y[agent_dir == RIGHT] - agent_view_size // 2

    # Facing down
    top_left[agent_dir == DOWN, 0] = agent_x[agent_dir == DOWN] - agent_view_size // 2
    top_left[agent_dir == DOWN, 1] = agent_y[agent_dir == DOWN]

    # Facing left
    top_left[agent_dir == LEFT, 0] = agent_x[agent_dir == LEFT] - agent_view_size + 1
    top_left[agent_dir == LEFT, 1] = agent_y[agent_dir == LEFT] - agent_view_size // 2

    # Facing up
    top_left[agent_dir == UP, 0] = agent_x[agent_dir == UP] - agent_view_size // 2
    top_left[agent_dir == UP, 1] = agent_y[agent_dir == UP] - agent_view_size + 1

    return top_left


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
