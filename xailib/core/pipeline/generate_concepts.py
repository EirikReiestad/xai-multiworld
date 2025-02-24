import logging
import os
import threading
from typing import Dict

from multiworld.base import MultiWorldEnv
from multiworld.multigrid.base import MultiGridEnv
from multiworld.multigrid.utils.wrappers import MultiGridConceptObsWrapper
from utils.common.model_artifact import ModelArtifact
from xailib.utils.pipeline_utils import create_model


def generate_concepts(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    concept_folder = os.path.join(config["path"]["concepts"])
    concepts = config["concepts"]
    all_concepts = []
    for concept in concepts:
        all_concepts.append(concept)
        all_concepts.append("negative_" + concept)

    if config["force_update"] is False:
        try:
            existing_concepts = os.listdir(concept_folder)
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
        target=generate_concepts_thread, args=(config, env, artifact)
    )
    thread.start()
    thread.join()


def generate_concepts_thread(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    if not isinstance(env, MultiGridEnv):
        raise ValueError("Only MultiGridEnv is supported for concept generation.")

    method = config["generate_concepts"]["method"]
    concepts = config["concepts"]
    observations = config["generate_concepts"]["observations"]
    eval = method == "policy"

    env_wrapped = MultiGridConceptObsWrapper(
        env,
        observations=observations,
        concepts=concepts,
        method=method,
    )

    model = create_model(config, artifact, env_wrapped, eval=eval)

    while True:
        model.learn()
