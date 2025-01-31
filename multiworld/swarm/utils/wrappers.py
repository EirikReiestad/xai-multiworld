import json
from typing import List, Literal

from multiworld.swarm.base import SwarmEnv
from multiworld.swarm.core.concept import get_concept_checks
from multiworld.utils.wrappers import ConceptObsWrapper
from utils.common.collections import DefaultEncoder


class SwarmConceptObsWrapper(ConceptObsWrapper):
    """
    Collect observations for a concept learning task.
    """

    def __init__(
        self,
        env: SwarmEnv,
        observations: int = 1000,
        concepts: List[str] | None = None,
        method: Literal["random", "policy"] = "policy",
        save_dir: str = "assets/concepts",
    ):
        super().__init__(
            env=env,
            observations=observations,
            concepts=concepts,
            method=method,
            save_dir=save_dir,
            concept_checks=get_concept_checks,
        )

    @property
    def encoder(self) -> json.JSONEncoder:
        return DefaultEncoder
