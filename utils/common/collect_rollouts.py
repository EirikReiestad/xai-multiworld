import logging
import os
import threading
from typing import Literal

from multiworld.base import MultiWorldEnv
from multiworld.utils.wrappers import ObservationCollectorWrapper
from utils.common.model import create_model
from utils.common.model_artifact import ModelArtifact
from utils.common.observation import Observation, observations_from_file


def collect_rollouts(
    env: MultiWorldEnv,
    artifact: ModelArtifact,
    observation_path: str = "assets/observations",
    force_update: bool = False,
) -> Observation:
    if force_update is False:
        try:
            observation = os.listdir(observation_path)
            if "observations.json" in set(observation):
                logging.info(
                    "Observations already exists, so we do not need to create them:)"
                )
                observations = observations_from_file(
                    os.path.join(observation_path, "observations.json")
                )
                return observations
        except FileNotFoundError:
            pass

    env._max_steps = int(env._width * 1.5)

    thread = threading.Thread(target=collect_rollouts_thread, args=(env, artifact))
    thread.start()
    thread.join()

    observations = observations_from_file(
        os.path.join(observation_path, "observations.json")
    )
    return observations


def collect_rollouts_thread(
    env: MultiWorldEnv,
    artifact: ModelArtifact,
    model_type: Literal["dqn"],
    observations: int,
    sample_rate: float,
    eval: bool,
    artifact_path: str = "artifacts/",
):
    env_wrapped = ObservationCollectorWrapper(
        env,
        observations,
        sample_rate,
    )

    model = create_model(artifact, model_type, artifact_path, env_wrapped, eval)

    while True:
        model.learn()
