import logging
import os
import threading
from typing import Dict

from multiworld.base import MultiWorldEnv
from multiworld.utils.wrappers import ObservationCollectorWrapper
from utils.common.model_artifact import ModelArtifact
from utils.common.observation import Observation, observations_from_file
from xailib.utils.pipeline_utils import create_model


def collect_rollouts(
    config: Dict, env: MultiWorldEnv, artifact: ModelArtifact
) -> Observation:
    rollout_folder = os.path.join(config["path"]["observations"])

    if config["force_update"] is False:
        try:
            observation = os.listdir(rollout_folder)
            if "observations.json" in set(observation):
                logging.info(
                    "Observations already exists, so we do not need to create them:)"
                )
                observations = observations_from_file(
                    os.path.join(config["path"]["observations"], "observations.json")
                )
                return observations
        except FileNotFoundError:
            pass

    env._max_steps = int(env._width * 1.5)

    thread = threading.Thread(
        target=collect_rollouts_thread, args=(config, env, artifact)
    )
    thread.start()
    thread.join()

    observations = observations_from_file(
        os.path.join(config["path"]["observations"], "observations.json")
    )
    return observations


def collect_rollouts_thread(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    env_wrapped = ObservationCollectorWrapper(
        env,
        observations=config["collect_rollouts"]["observations"],
        sample_rate=config["collect_rollouts"]["sample_rate"],
    )

    eval = config["collect_rollouts"]["method"] == "policy"
    model = create_model(config, artifact, env_wrapped, eval=eval)

    while True:
        model.learn()
