import numpy as np
from numpy.typing import NDArray as ndarray

from multigrid.core.agent import AgentState
from multigrid.core.constants import Color, Direction, Type
from multigrid.core.world_object import WorldObject

UNSEEN_ENCODING = WorldObject(Type.unseen, Color.from_index(0)).encode()
ENCODE_DIM = WorldObject.dim

GRID_ENCODING_IDX = slice(None)

AGENT_DIR_IDX = AgentState.DIR
AGENT_POS_IDX = AgentState.POS
AGENT_TERMINATED_IDX = AgentState.TERMINATED
AGENT_CARRYING_IDX = AgentState.CARRYING
AGENT_ENCODING_IDX = AgentState.ENCODING

TYPE = int(Type.wall)
BOX = int(Type.box)

RIGHT = int(Direction.right)
LEFT = int(Direction.left)
UP = int(Direction.up)
DOWN = int(Direction.down)


def gen_obs_grid_encoding(
    grid_state: ndarray[np.int_], agent_state: ndarray[np.int_], agent_view_size: int
) -> ndarray[np.int_]:
    obs_grid = gen_obs_grid(grid_state, agent_state, agent_view_size)
    # Generate and apply visability mask
    vis_mask = get_vis_mask(obs_grid)
    num_agents = len(agent_state)
    for agent in range(num_agents):
        for i in range(agent_view_size):
            for j in range(agent_view_size):
                if not vis_mask[agent, i, j]:
                    obs_grid[agent, i, j] = UNSEEN_ENCODING
    return obs_grid


def gen_obs_grid(
    grid_state: ndarray[np.int_], agent_state: ndarray[np.int_], agent_view_size: int
) -> ndarray[np.int_]:
    num_agents = len(agent_state)
    obs_width, obs_height = agent_view_size, agent_view_size

    # Process agent states
    agent_grid = agent_state[..., AGENT_ENCODING_IDX]
    agent_dir = agent_state[..., AGENT_DIR_IDX]
    agent_pos = agent_state[..., AGENT_POS_IDX]
    agent_terminated = agent_state[..., AGENT_TERMINATED_IDX]
    agent_carrying = agent_state[..., AGENT_CARRYING_IDX]

    if num_agents > 1:
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

    top_left = get_view_exts(agent_dir, agent_pos, agent_view_size)
    topX, topY = top_left[:, 0], top_left[:, 1]

    # Population observation grid
    num_left_rotations = (agent_dir + 1) % 4
    obs_grid = np.empty((num_agents, obs_width, obs_height, ENCODE_DIM), dtype=np.int_)
    for agent in range(num_agents):
        for i in range(obs_width):
            for j in range(obs_height):
                x, y = topX[agent] - i, topY[agent] - j
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
                    obs_grid[agent, i_rot, j_rot] = grid_encoding[x, y]
                else:
                    obs_grid[agent, i_rot, j_rot] = UNSEEN_ENCODING
    obs_grid[:, obs_width // 2, obs_height - 1] = agent_carrying
    return obs_grid


def get_vis_mask(obs_grid: ndarray[np.int_]) -> ndarray[np.bool_]:
    num_agents, obs_width, obs_height, _ = obs_grid.shape
    vis_mask = np.zeros((num_agents, obs_width, obs_height), dtype=np.bool_)
    vis_mask[:, obs_width // 2, obs_height - 1] = True  # Agent relative position
    for agent in range(num_agents):
        for j in range(obs_height - 1, -1, -1):
            # Forward pass
            for i in range(obs_width - 1):
                if vis_mask[agent, i, j]:
                    vis_mask[agent, i + 1, j] = True
                    if j > 0:
                        vis_mask[agent, i + 1, j - 1] = True
                        vis_mask[agent, i, j - 1] = True
            # Backward pass
            for i in range(obs_width - 1, 0, -1):
                if vis_mask[agent, i, j]:
                    vis_mask[agent, i - 1, j] = True
                    if j > 0:
                        vis_mask[agent, i - 1, j - 1] = True
                        vis_mask[agent, i, j - 1] = True
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
