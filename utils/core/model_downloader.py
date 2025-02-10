import shutil
import traceback

import torch.nn as nn

from utils.core.model_loader import ModelLoader
from utils.core.wandb import WandB


class ModelDownloader(WandB):
    def __init__(
        self,
        project_folder: str,
        model_name: str,
        models: list[str],
        model: nn.Module | None = None,
        model_folder: str = "artifacts",
        folder_suffix: str = "",
    ):
        self.wandb = WandB(
            project=project_folder,
            run_name=None,
            reinit=True,
            save_steps=None,
            tags=[],
            dir=".",
            only_api=True,
        )

        self._clean_models(model_folder)

        self._model_name = model_name
        self._run_id = f"eirikreiestad-ntnu/{project_folder}"
        self._extract_model_names(models)

        self._model = model
        self.folder_suffix = folder_suffix

    def _clean_models(self, path: str):
        shutil.rmtree(path, ignore_errors=True)

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
        artifact = ModelLoader.load_model_artifact_from_path(artifact_dir)
        model = ModelLoader.load_model_from_artifact(artifact, self._model)
