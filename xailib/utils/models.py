from typing import Dict

from multiworld.base import MultiWorldEnv
from utils.common.model_artifact import ModelArtifact
from xailib.utils.misc import create_model


def get_models(config: Dict, env: MultiWorldEnv, artifact: ModelArtifact):
    model = create_model(config, artifact, env, eval=True)

    models = ModelLoader.load_models_from_path("artifacts", model.model)
    return models
