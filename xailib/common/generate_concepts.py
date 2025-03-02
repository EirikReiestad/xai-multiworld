import logging
import os
import threading
from typing import List, Literal

from multiworld.base import MultiWorldEnv
from multiworld.multigrid.base import MultiGridEnv
from multiworld.multigrid.utils.wrappers import MultiGridConceptObsWrapper
from utils.common.model import create_model
from utils.common.model_artifact import ModelArtifact


def generate_concepts(
    concepts: List[str],
    env: MultiWorldEnv,
    observations: int,
    artifact: ModelArtifact,
    method: Literal["random", "policy"],
    model_type: Literal["dqn"],
    force_update: bool = False,
    artifact_path: str = os.path.join("artifacts"),
    concept_path: str = os.path.join("assets", "concepts"),
    result_dir: str = os.path.join("assets", "results"),
):
    all_concepts = []
    for concept in concepts:
        all_concepts.append(concept)
        all_concepts.append("negative_" + concept)

    if force_update is False:
        try:
            existing_concepts = os.listdir(concept_path)
            existing_concepts = (
                [existing_concepts]
                if isinstance(existing_concepts, str)
                else existing_concepts
            )
            existing_concepts = [
                word.rstrip(".json") if word.endswith(".json") else word
                for word in existing_concepts
            ]

            if set(all_concepts).issubset(set(existing_concepts)):
                logging.info(
                    "Concepts already exists, so we do not need to download them:)"
                )
                return
        except FileNotFoundError:
            pass

    thread = threading.Thread(
        target=generate_concepts_thread,
        args=(
            concepts,
            env,
            observations,
            artifact,
            method,
            model_type,
            artifact_path,
            result_dir,
        ),
    )
    thread.start()
    thread.join()


def generate_concepts_thread(
    concepts: List[str],
    env: MultiWorldEnv,
    observations: int,
    artifact: ModelArtifact,
    method: Literal["random", "policy"],
    model_type: Literal["dqn"],
    artifact_path: str = os.path.join("artifacts"),
    results_dir: str = os.path.join("assets", "results"),
):
    if not isinstance(env, MultiGridEnv):
        raise ValueError("Only MultiGridEnv is supported for concept generation.")

    eval = method == "policy"

    env_wrapped = MultiGridConceptObsWrapper(
        env,
        observations=observations,
        concepts=concepts,
        method=method,
        results_dir=results_dir,
    )

    model = create_model(artifact, model_type, artifact_path, env_wrapped, eval)

    while True:
        model.learn()
