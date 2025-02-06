from typing import Callable, Dict, List

import numpy as np
from numpy.typing import NDArray

from multiworld.multigrid.core.constants import WorldObjectType
from multiworld.multigrid.core.world_object import WorldObject


def _random_observation_concept(_: NDArray[np.int_]) -> bool:
    rand_float = np.random.uniform()
    if rand_float < 0.1:
        return True
    return False


def _random_negative_observation_concept(_: NDArray[np.int_]) -> bool:
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
        for cell in row[row.shape[0] // 2 :]:
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
        for cell in row[0 : row.size // 2]:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.goal.to_index():
                return True
    return False


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
def _agent_to_right_concept(view: Dict[str, NDArray[np.int_]]) -> bool:
    observation = view["observation"]
    dir = view["direction"]

    for row in observation:
        for cell in row[row.shape[0] // 2 :]:
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
        for cell in row[0 : row.size // 2]:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.agent.to_index():
                return True
    return False


concept_checks: Dict[str, Callable] = {
    "random": _random_observation_concept,
    "random_negative": _random_observation_concept,
    "goal_in_view": _goal_in_view_concept,
    "goal_to_right": _goal_to_right_concept,
    "goal_to_left": _goal_to_left_concept,
    "goal_in_front": _goal_in_front_concept,
    "agent_in_view": _agent_in_view_concept,
    # "wall_in_view": _wall_in_view_concept,
    "agent_to_right": _agent_to_right_concept,
    "agent_to_left": _agent_to_left_concept,
    "agent_in_front": _agent_in_front_concept,
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
        concept_bitmaps.extend(_get_agent_concept_values(state))
        # Initialize a list to store concept values (0 or 1) for the current state
    # Return the list of concept bit maps

    return concept_bitmaps


def _get_agent_concept_values(state: Dict[str, Dict[str, NDArray]]) -> List:
    concept_values = []

    for agent_state in state.values():
        # For each concept, check its presence (using concept checks, as defined in `get_concept_checks`)
        concept_values.append(_get_concept_check_bitmap(agent_state))

    return concept_values


def _get_concept_check_bitmap(state: Dict[str, NDArray]) -> List[NDArray]:
    if state is None:
        return [0] * len(concept_checks)
    concept_bitmap = []
    for concept_check_name, concept_check in concept_checks.items():
        # Apply the concept check to the state (assuming it returns True/False)
        concept_present = concept_check(
            state
        )  # Will return True if concept is present, else False
        concept_bitmap.append(1 if concept_present else 0)

    # Convert the list of 0s and 1s to a numpy array for the current state concept_bit_maps.append(np.array(concept_values))
    return concept_bitmap
