from typing import Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

from multiworld.multigrid.core.constants import Direction, WorldObjectType
from multiworld.multigrid.core.world_object import WorldObject
from multiworld.utils.typing import ObsType


def _random_observation_concept(_: NDArray[np.int_]) -> bool:
    rand_float = np.random.uniform()
    if rand_float < 0.1:
        return True
    return False


@staticmethod
def _goal_in_view_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    dir = view["direction"]

    for row in observation:
        for cell in row:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.goal.to_index():
                return True
    return False


@staticmethod
def _goal_to_right_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    dir = view["direction"]

    for row in observation:
        for cell in row[row.shape[0] // 2 + 1 :]:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.goal.to_index():
                return True
    return False


@staticmethod
def _goal_to_left_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    dir = view["direction"]

    for row in observation:
        for cell in row[0 : row.shape[0] // 2]:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.goal.to_index():
                return True
    return False


@staticmethod
def _goal_in_front_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    dir = view["direction"]

    for row in observation[1:]:
        cell = row[row.shape[0] // 2]
        type_idx = cell[WorldObject.TYPE]
        if type_idx == WorldObjectType.goal.to_index():
            return True
    return False


@staticmethod
def _goal_at_top_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]

    width, height = observation.shape[:2]
    for row in observation[: height // 2]:
        for cell in row:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.goal.to_index():
                return True
    return False


@staticmethod
def _goal_around_middle_front_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    for row in observation[1:]:
        for cell in row[row.shape[0] // 2 - 1 : row.shape[0] // 2 + 2]:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.goal.to_index():
                return True
    return False


@staticmethod
def _goal_at_bottom_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]

    width, height = observation.shape[:2]
    for row in observation[height // 2 :]:
        for cell in row:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.goal.to_index():
                return True
    return False


@staticmethod
def _next_to_goal_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    # NOTE: This only works where we have environment that is surrouned by walls and no other placed walls
    observation = view["observation"]
    width, height = observation.shape[:2]
    left_cell = observation[height - 1][width // 2 - 1]
    right_cell = observation[height - 1][width // 2 + 1]
    upper_cell = observation[height - 2][width // 2]
    return (
        left_cell[WorldObject.TYPE] == WorldObjectType.goal.to_index()
        or right_cell[WorldObject.TYPE] == WorldObjectType.goal.to_index()
        or upper_cell[WorldObject.TYPE] == WorldObjectType.goal.to_index()
    )


@staticmethod
def _agent_in_view_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    dir = view["direction"]

    for row in observation:
        for cell in row:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.agent.to_index():
                return True
    return False


@staticmethod
def _agent_to_right_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    dir = view["direction"]

    for row in observation:
        for cell in row[row.shape[0] // 2 + 1 :]:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.agent.to_index():
                return True
    return False


@staticmethod
def _agent_to_left_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    dir = view["direction"]

    for row in observation:
        for cell in row[0 : row.shape[0] // 2]:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.agent.to_index():
                return True
    return False


@staticmethod
def _agent_in_front_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    dir = view["direction"]

    for row in observation[1:]:
        cell = row[row.shape[0] // 2]
        type_idx = cell[WorldObject.TYPE]
        if type_idx == WorldObjectType.agent.to_index():
            return True
    return False


@staticmethod
def _rotated_right(view: Dict[str, NDArray[np.int_]]) -> bool:
    dir_idx = np.argmax(view["direction"])
    return Direction.right == dir_idx


@staticmethod
def _rotated_left(view: Dict[str, NDArray[np.int_]]) -> bool:
    dir_idx = np.argmax(view["direction"])
    return Direction.left == dir_idx


@staticmethod
def _rotated_up(view: Dict[str, NDArray[np.int_]]) -> bool:
    dir_idx = np.argmax(view["direction"])
    return Direction.up == dir_idx


@staticmethod
def _rotated_down(view: Dict[str, NDArray[np.int_]]) -> bool:
    dir_idx = np.argmax(view["direction"])
    return Direction.down == dir_idx


@staticmethod
def _wall_in_view_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    dir = view["direction"]

    for row in observation:
        for cell in row:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.wall.to_index():
                return True
    return False


@staticmethod
def _wall_in_front_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    # NOTE: This only works where we have environment that is surrouned by walls and no other placed walls
    observation = view["observation"]
    width, height = observation.shape[:2]
    left_cell = observation[0][0]
    right_cell = observation[0][width - 1]
    return (
        left_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
        and right_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
    )


@staticmethod
def _wall_to_right_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    # NOTE: This only works where we have environment that is surrouned by walls and no other placed walls
    observation = view["observation"]
    width, height = observation.shape[:2]
    upper_cell = observation[0][width - 1]
    lower_cell = observation[height - 1][width - 1]
    return (
        upper_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
        and lower_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
    )


@staticmethod
def _wall_to_left_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    # NOTE: This only works where we have environment that is surrouned by walls and no other placed walls
    observation = view["observation"]
    width, height = observation.shape[:2]
    upper_cell = observation[0][0]
    lower_cell = observation[height - 1][0]
    return (
        upper_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
        and lower_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
    )


@staticmethod
def _next_to_wall_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    # NOTE: This only works where we have environment that is surrouned by walls and no other placed walls
    observation = view["observation"]
    width, height = observation.shape[:2]
    left_cell = observation[height - 1][width // 2 - 1]
    right_cell = observation[height - 1][width // 2 + 1]
    upper_cell = observation[height - 2][width // 2]
    return (
        left_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
        or right_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
        or upper_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
    )


@staticmethod
def _close_to_wall_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    # NOTE: This only works where we have environment that is surrouned by walls and no other placed walls
    observation = view["observation"]
    width, height = observation.shape[:2]
    left_cell = observation[height - 1][width // 4 - 1]
    right_cell = observation[height - 1][1 - width // 4]
    upper_cell = observation[1 - height // 4][width // 2]
    return (
        left_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
        or right_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
        or upper_cell[WorldObject.TYPE] == WorldObjectType.wall.to_index()
    )


concept_checks: Dict[str, Callable] = {
    "random": _random_observation_concept,
    "goal_in_view": _goal_in_view_concept,
    "goal_to_right": _goal_to_right_concept,
    "goal_to_left": _goal_to_left_concept,
    "goal_in_front": _goal_in_front_concept,
    "goal_at_top": _goal_at_top_concept,
    "goal_at_bottom": _goal_at_bottom_concept,
    "goal_around_middle_front": _goal_around_middle_front_concept,
    "next_to_goal": _next_to_goal_concept,
    "agent_in_view": _agent_in_view_concept,
    "agent_to_right": _agent_to_right_concept,
    "agent_to_left": _agent_to_left_concept,
    "agent_in_front": _agent_in_front_concept,
    "rotated_right": _rotated_right,
    "rotated_left": _rotated_left,
    "rotated_up": _rotated_up,
    "rotated_down": _rotated_down,
    "wall_in_view": _wall_in_view_concept,
    "wall_in_front": _wall_in_front_concept,
    "wall_to_right": _wall_to_right_concept,
    "wall_to_left": _wall_to_left_concept,
    "next_to_wall": _next_to_wall_concept,
    "close_to_wall": _close_to_wall_concept,
}


def get_concept_checks(concepts: list[str] | None):
    return (
        concept_checks
        if concepts is None
        else {key: value for key, value in concept_checks.items() if key in concepts}
    )


def get_concept_values(states: List[Dict[str, Dict[str, NDArray]]]) -> List[List[int]]:
    """
    This function processes the next state observations and checks for the presence of each concept.
    It returns a bit map (list of 0 or 1) for each concept, indicating whether it's present in the given state.

    Args:
        state(List[Dict[str, Dict[str, NDArray]]]): List of state observations for multiple agents.
        batch size * num agents

    Returns:
        List[NDArray]: List of concept bit maps, each representing the presence (1) or absence (0) of concepts.
    """
    # List to store the bit maps for each concept in the batch
    concept_bitmaps = []

    # Iterate over the states for each agent
    for state in states:
        concept_bitmaps.extend(get_observation_concept_values(state))
        # Initialize a list to store concept values (0 or 1) for the current state
    # Return the list of concept bit maps

    return concept_bitmaps


def get_observation_concept_values(state: Dict[str, Dict[str, NDArray]]) -> List:
    concept_values = []

    for agent_state in state.values():
        # For each concept, check its presence (using concept checks, as defined in `get_concept_checks`)
        concept_values.append(
            get_concept_check_bitmap(agent_state, list(concept_checks.keys()))
        )

    return concept_values


def get_concept_check_bitmap(
    state: Dict[str, NDArray], concepts: List[str]
) -> List[NDArray]:
    concept_bitmap = []

    for concept in concepts:
        # Apply the concept check to the state (assuming it returns True/False)
        concept_present = concept_checks[concept](
            state
        )  # Will return True if concept is present, else False
        concept_bitmap.append(int(concept_present))

    # Convert the list of 0s and 1s to a numpy array for the current state concept_bit_maps.append(np.array(concept_values))
    return concept_bitmap
