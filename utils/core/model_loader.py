from typing import Mapping, Any, List, Dict
import os


class ModelLoader:
    @staticmethod
    def load_from_path(path: str) -> Dict[str, Mapping[str, Any]]:
        models = {}

        for model_dir in os.listdir(path):
            model_dir_path = os.path.join(path, model_dir)
            if not os.path.isdir(model_dir_path):
                continue
            model_file = None
            metadata_file = None

            model_path = os.path.join(model_dir_path, k")

            for file in os.listdir(model_dir_path):
                file_path = os.path.join(model_dir_path, file)

                if file.endswith(".pth"):
                    model_file = file_path
                elif file.endswith("metadata.json"):
                    metadata_file = file_path

            if not model_file or not metadata_file:
                raise ValueError(f"Model and metadata files not found in {model_dir_path}")

            model = torch.load(model_file)
            model.eval()

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

