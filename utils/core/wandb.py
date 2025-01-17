from enum import Enum
import json
import logging
import os
from abc import ABC
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

import wandb
from rllib.utils.image import save_gif


class LogMethod(Enum):
    OVERWRITE = "overwrite"
    CUMULATIVE = "cumulative"
    AVERAGE = "average"


class WandB(ABC):
    def __init__(
        self,
        project: str | None,
        run_name: str | None,
        reinit: bool | None,
        save_steps: int | None,
        tags: list[str] | None,
        dir: str | None,
        only_api: bool = False,
    ):
        if project is None:
            self._api = None
            return

        self._api = wandb.Api()

        if only_api:
            return

        wandb.init(
            project=project,
            name=run_name,
            reinit=reinit,
            tags=tags,
            dir=dir,
        )
        self._save_steps = save_steps

        self._log = {}
        self._log_average = defaultdict(list)
        self._artifact = None
        self._frames = []

    def log_frame(self, frame: Optional[np.ndarray], step: int = 0):
        """
        Logging a frame, a rendering of the environment.
        """
        if self._api is None:
            return
        if frame is None:
            return
        if self._save_steps is not None and step % self._save_steps != 0:
            return
        assert isinstance(
            frame, np.ndarray
        ), f"Frame must be a numpy array, but got {frame}"
        self._frames.append(frame)

    def log_model(self, model: nn.Module, model_name: str = "model", step: int = 0):
        if self._api is None:
            return
        if self._save_steps is not None and step % self._save_steps != 0:
            return
        file_path = f"{model_name}.pth"
        torch.save(model.state_dict(), file_path)

        self._artifact = wandb.Artifact(model_name, type="model")
        self._artifact.add_file(file_path)

        os.remove(file_path)

    def log(self, data: dict):
        if self._api is None:
            return
        if self._artifact is not None:
            wandb.log_artifact(self._artifact)
        wandb.log(data)

    def add_log(
        self,
        key: str,
        value: float,
        method: LogMethod = LogMethod.OVERWRITE,
    ):
        if self._api is None:
            return

        if method == LogMethod.OVERWRITE:
            self._log[key] = value
        elif method == LogMethod.CUMULATIVE:
            self._log[key] = self._log.get(key, 0) + value
        elif method == LogMethod.AVERAGE:
            self._log_average[key].append(value)
        else:
            raise ValueError(f"Method {method} not recognized.")

    def download_model(
        self, run_path: str, model_artifact: str, version_number: str
    ) -> tuple[None | str, None | dict]:
        if self._api is None:
            return None, None

        if version_number == "":
            logging.error("Error: version_number cannot be empty")
        artifact_path = f"{run_path}/{model_artifact}:{version_number}"
        try:
            artifact = self._api.artifact(
                artifact_path,
            )
            artifact_dir = artifact.download()
            logging.info("model downloaded at: " + artifact_dir)
            logging.info("Metadata: " + str(artifact.metadata))
            metadata_path = os.path.join(artifact_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(artifact.metadata, f)
            logging.info("Metadata saved at: " + metadata_path)
            return artifact_dir, artifact.metadata
        except wandb.Error as e:
            logging.error(f"Error: Could not load model with artifact: {artifact_path}")
            logging.error(f"Error: {e}")
            return None, None

    def commit_log(self):
        if self._api is None:
            return

        print("test")

        self._commit_frames()

        for key, values in self._log_average.items():
            self._log[key] = np.mean(values)

        self.log(self._log)
        self._log.clear()
        self._log_average.clear()
        self._artifact = None

    def _commit_frames(self):
        if len(self._frames) == 0:
            return
        gif_path = "assets/temp.gif"
        save_gif(self._frames, gif_path)
        try:
            gif = wandb.Image(gif_path)
            self._log["gif"] = gif
        except Exception as e:
            logging.error(f"Error: Could not save gif: {e}")
        self._frames = []
