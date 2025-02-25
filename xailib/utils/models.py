from typing import Dict

from multiworld.base import MultiWorldEnv
from utils.common.model_artifact import ModelArtifact
from xailib.utils.misc import create_model


def get_models(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    model = create_model(config, artifact, env, eval=True)

    models = ModelLoader.load_models_from_path("artifacts", model.model)
    return models


def download_models(config: Dict):
    models = [
        f"model_{i}:latest"
        for i in range(
            config["wandb"]["models"]["low"],
            config["wandb"]["models"]["high"],
            config["wandb"]["models"]["step"],
        )
    ]
    model_folder = os.path.join(config["path"]["artifacts"])

    if config["force_update"] is False:
        try:
            artifacts = [
                model_name.split(":")[0] for model_name in os.listdir(model_folder)
            ]
            model_names = [model_name.split(":")[0] for model_name in models]

            if sorted(artifacts) == sorted(model_names):
                logging.info(
                    "Artifacts already exists, so we do not need to download them:)"
                )
                return
        except FileNotFoundError:
            pass

    model_downloader = ModelDownloader(
        project_folder=config["wandb"]["project_folder"],
        models=models,
        model_folder=model_folder,
    )
    model_downloader.download()
