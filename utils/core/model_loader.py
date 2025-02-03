import copy
import json
import os
import re
from typing import Dict

import torch
import torch.nn as nn

from utils.common.model_artifact import ModelArtifact


class ModelLoader:
    @staticmethod
    def load_models_from_path(
        path: str, network: nn.Module
    ) -> Dict[str, ModelArtifact]:
        models = {}
        model_dirs = os.listdir(path)
        sorted_model_dirs = sorted(
            model_dirs, key=lambda x: int(re.search(r"\d+", x).group())
        )
        for model_dir in sorted_model_dirs:
            model_dir_path = os.path.join(path, model_dir)
            if not os.path.isdir(model_dir_path):
                continue
            model_artifact = ModelLoader.load_model_from_path(
                model_dir_path, copy.deepcopy(network)
            )
            models[model_dir] = model_artifact
        return models

    @staticmethod
    def load_latest_model_from_path(
        path: str, network: nn.Module
    ) -> Dict[str, ModelArtifact]:
        model_dirs = os.listdir(path)
        sorted_model_dirs = sorted(
            model_dirs, key=lambda x: int(re.search(r"\d+", x).group())
        )
        latest_model_dir = sorted_model_dirs[-1]
        latest_model_dir_path = os.path.join(path, latest_model_dir)
        return {
            "latest": ModelLoader.load_model_from_path(latest_model_dir_path, network)
        }

    @staticmethod
    def load_model_from_path(path: str, network: nn.Module) -> ModelArtifact:
        model_file = None
        metadata_file = None

        for file in os.listdir(path):
            file_path = os.path.join(path, file)

            if file.endswith(".pth"):
                model_file = file_path
            elif file.endswith("metadata.json"):
                metadata_file = file_path

        if not model_file or not metadata_file:
            raise ValueError(f"Model and metadata files not found in {path}")

        model_weights = torch.load(model_file, weights_only=True)

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        model = None
        try:
            network.load_state_dict(model_weights)
            model = network
        except AttributeError as e:
            raise AttributeError(f"Model class does not match network: {e}")

        model_artifact = ModelArtifact(
            model_weights=model_weights, metadata=metadata, model=model
        )
        return model_artifact
