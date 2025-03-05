import json
from typing import List, Literal

from multiworld.multigrid.base import MultiGridEnv
from multiworld.multigrid.core.concept import get_concept_checks
from multiworld.utils.wrappers import ConceptObsWrapper
from utils.common.numpy_collections import NumpyEncoder


class MultiGridConceptObsWrapper(ConceptObsWrapper):
    """
    Collect observations for a concept learning task.
    """

    def __init__(
        self,
        env: MultiGridEnv,
        observations: int = 1000,
        concepts: List[str] | None = None,
        method: Literal["random", "policy"] = "policy",
        save_dir: str = "assets/concepts",
        results_dir: str = "assets/results",
    ):
        super().__init__(
            env=env,
            observations=observations,
            concepts=concepts,
            method=method,
            save_dir=save_dir,
            concept_checks=get_concept_checks,
            result_save_dir=results_dir,
        )

    @property
    def encoder(self) -> json.JSONEncoder:
        return NumpyEncoder
