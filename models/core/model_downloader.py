import getpass
import json
import os
import traceback
import shutil

import torch

from managers import WandBConfig, WandBManager
from rl.src.dqn.policies import DQNPolicy


class ModelDownloader(WandB):
    def __init__(
        self,
        model: DQNPolicy,
        project_folder: str,
        model_name: str,
        models: list[str],
        folder_suffix: str = "",
    ):
        current_user = getpass.getuser()
        project = f"{current_user}"
        wandb_config = WandBConfig(project=project)
        self.wandb_manager = WandBManager(active=False, config=wandb_config)

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
        artifact_dir, metadata = self.wandb_manager.load_model(
            self._run_id, model_artifact, version_number
        )
        if artifact_dir is None or metadata is None:
            raise Exception(f"Model not found, {traceback.format_exc}")
        self._metadata = metadata

        path = f"{artifact_dir}/{self._model_name}"
        if not path.endswith(".pt"):
            path += ".pt"

        self._model.policy_net.load_state_dict(torch.load(path, weights_only=True))
        self._model.target_net.load_state_dict(self._model.policy_net.state_dict())
        self._model.policy_net.eval()
        self._model.target_net.eval()

        self._save_locally(model_artifact, version_number, metadata)

    def _save_locally(self, model_artifact: str, version_number: str, metadata: dict):
        folder_path = os.path.join(
            "models", "latest" + self.folder_suffix, model_artifact
        )
        os.makedirs(folder_path, exist_ok=True)

        path = os.path.join(folder_path, version_number)
        path += ".pt"

        torch.save(self._model.policy_net.state_dict(), path)

        folder_path = os.path.join(
            "models", "metadata" + self.folder_suffix, model_artifact
        )
        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, version_number)
        path += ".json"

        json.dump(metadata, open(path, "w"))
