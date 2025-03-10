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
    n: int,
    method: Literal["policy", "random"],
    observation_path: str = os.path.join("assets", "observations"),
    force_update: bool = False,
    model_type: Literal["dqn"] = "dqn",
    sample_rate: float = 1.0,
    artifact_path: str = os.path.join("artifacts"),
) -> Observation:
    if not os.path.exists(observation_path):
        os.mkdir(observation_path)
    if force_update is False:
        try:
            observation = os.listdir(observation_path)
            if "observations.json" in set(observation):
                observations = observations_from_file(
                    os.path.join(observation_path, "observations.json")
                )
                logging.info(
                    f"Observations already exists, so we do not need to create them:) Observation size: {len(observations)}"
                )
                return observations
        except FileNotFoundError:
            pass

    env._max_steps = int(env._width * 1.5)

    eval = method == "policy"
    thread = threading.Thread(
        target=collect_rollouts_thread,
        args=(
            env,
            artifact,
            model_type,
            n,
            sample_rate,
            eval,
            observation_path,
            artifact_path,
        ),
    )
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
    directory: str,
    artifact_path: str = "artifacts/",
):
    env_wrapped = ObservationCollectorWrapper(
        env,
        observations,
        sample_rate,
        directory=directory,
    )

    model = create_model(artifact, model_type, artifact_path, env_wrapped, eval)

    while True:
        model.learn()
