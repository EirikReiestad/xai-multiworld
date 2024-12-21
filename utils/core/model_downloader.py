import getpass
import json
import os
import traceback
import shutil

import torch

from utils.core.wandb import WandB
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.algorithm import Algorithm
import torch.nn as nn
from utils.core.model_loader import ModelLoader


class ModelDownloader(WandB):
    def __init__(
        self,
        project_folder: str,
        model_name: str,
        models: list[str],
        model: nn.Module | None = None,
        folder_suffix: str = "",
    ):
        self.wandb = WandB(
            project=project_folder,
            run_name=None,
            reinit=True,
            tags=[],
            dir=".",
            only_api=True,
        )

        self._model_name = model_name
        self._run_id = f"eirikreiestad-ntnu/{project_folder}"
        self._extract_model_names(models)

        self._model = model
        self.folder_suffix = folder_suffix

        self._clean_models()

    def _clean_models(self):
        model_folder = os.path.join("models", "latest" + self.folder_suffix)
        shutil.rmtree(model_folder, ignore_errors=True)
        metadata_folder = os.path.join("models", "metadata" + self.folder_suffix)
        shutil.rmtree(metadata_folder, ignore_errors=True)

    def _extract_model_names(self, model_names: list[str]):
        model_artifacts = []
        version_numbers = []

        for model_name in model_names:
            split = model_name.split(":")
            model_artifacts.append(split[0])
            version_numbers.append(split[1])

        self._model_artifacts = model_artifacts
        self._version_numbers = version_numbers

    def download(self):
        for model_idx in range(len(self._model_artifacts)):
            model_artifact = self._model_artifacts[model_idx]
            version_number = self._version_numbers[model_idx]
            self._load_model(model_artifact, version_number)

    def _load_model(self, model_artifact: str, version_number: str):
        artifact_dir, metadata = self.wandb.download_model(
            self._run_id, model_artifact, version_number
        )
        if artifact_dir is None or metadata is None:
            raise Exception(f"Model not found, {traceback.format_exc}")

        if self._model is None:
            return

        # Test if model is loaded correctly
        model_artifacts = ModelLoader.load_model_from_path(artifact_dir, self._model)
