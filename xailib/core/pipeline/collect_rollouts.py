import threading
from typing import Dict

from multiworld.base import MultiWorldEnv
from multiworld.multigrid.base import MultiGridEnv
from multiworld.multigrid.utils.wrappers import MultiGridConceptObsWrapper
from multiworld.swarm.base import SwarmEnv
from multiworld.utils.wrappers import ObservationCollectorWrapper
from utils.common.model_artifact import ModelArtifact
from xailib.utils.pipeline_utils import create_model


def collect_rollouts(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    thread = threading.Thread(
        target=collect_rollouts_thread, args=(config, env, artifact)
    )
    thread.start()
    thread.join()


def collect_rollouts_thread(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    env_wrapped = ObservationCollectorWrapper(
        env,
        observations=config["collect_rollouts"]["observations"],
        sample_rate=config["collect_rollouts"]["sample_rate"],
    )

    model = create_model(config, artifact, env_wrapped, eval=False)

    while True:
        model.learn()


def generate_concepts(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    thread = threading.Thread(
        target=generate_concepts_thread, args=(config, env, artifact)
    )
    thread.start()
    thread.join()


def generate_concepts_thread(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    if not isinstance(env, MultiGridEnv):
        raise ValueError("Only MultiGridEnv is supported for concept generation.")

    method = config["generate_concepts"]["method"]
    concepts = config["generate_concepts"]["concepts"]
    observations = config["generate_concepts"]["observations"]

    env_wrapped = MultiGridConceptObsWrapper(
        env,
        observations=observations,
        concepts=concepts,
        method=method,
    )

    model = create_model(config, artifact, env_wrapped, eval=True)

    while True:
        model.learn()
