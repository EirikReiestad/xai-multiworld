import logging
import os
import threading
from typing import Dict

from multiworld.base import MultiWorldEnv
from multiworld.utils.wrappers import ObservationCollectorWrapper
from utils.common.model_artifact import ModelArtifact
from xailib.utils.pipeline_utils import create_model


def collect_rollouts(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    rollout_folder = os.path.join(config["path"]["observations"])

    if config["force_update"] is False:
        try:
            observation = os.listdir(rollout_folder)
            if "observations.json" in set(observation):
                logging.info(
                    "Observations already exists, so we do not need to create them:)"
                )
                return
        except FileNotFoundError:
            pass

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
