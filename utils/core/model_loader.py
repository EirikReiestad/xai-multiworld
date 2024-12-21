import json
import logging
import os
from typing import Any, Dict, List, Mapping

import torch
import torch.nn as nn

from utils.common.model_artifact import ModelArtifact


class ModelLoader:
    @staticmethod
    def load_from_path(
        path: str, network: nn.Module | None = None
    ) -> Dict[str, ModelArtifact]:
        models = {}

        for model_dir in os.listdir(path):
            model_dir_path = os.path.join(path, model_dir)
            if not os.path.isdir(model_dir_path):
                continue

            model_file = None
            metadata_file = None

            for file in os.listdir(model_dir_path):
                file_path = os.path.join(model_dir_path, file)

                if file.endswith(".pth"):
                    model_file = file_path
                elif file.endswith("metadata.json"):
                    metadata_file = file_path

            if not model_file or not metadata_file:
                raise ValueError(
                    f"Model and metadata files not found in {model_dir_path}"
                )

            model_weights = torch.load(model_file, weights_only=True)

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            model = None
            if network is not None:
                try:
                    network.load_state_dict(
                        torch.load(model_weights, weights_only=True)
                    )
                    model = network
                except Exception as e:
                    logging.warning(f"Failed to load model with network: {e}")

            model_artifact = ModelArtifact(
                model_weights=model_weights, metadata=metadata, model=model
            )
            models[model_dir] = model_artifact
        return models
