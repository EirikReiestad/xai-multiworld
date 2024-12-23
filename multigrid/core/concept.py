import numpy as np
from numpy.typing import NDArray
from multigrid.core.constants import WorldObjectType
from multigrid.core.world_object import WorldObject


def _random_observation_concept(self, _: NDArray[np.int_]) -> bool:
    rand_float = np.random.uniform()
    if rand_float < 0.2:
        return True
    return False


@staticmethod
def _goal_in_view_concept(view: NDArray[np.int_]) -> bool:
    for row in view:
        for cell in row:
            type_idx = cell[WorldObject.TYPE]
            if type_idx == WorldObjectType.goal.to_index():
                return True
    return False


concept_checks = {
    "random": _random_observation_concept,
    "goal": _goal_in_view_concept,
}


def get_concept_checks(concepts: list[str] | None):
    return (
        concept_checks
        if concepts is None
        else {key: value for key, value in concept_checks.items() if key in concepts}
    )
