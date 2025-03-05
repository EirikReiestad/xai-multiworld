import os

import numpy as np
import shap

from rllib.utils.torch.processing import observations_seperate_to_torch
from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.common.observation import (
    Observation,
    filter_observations,
    normalize_observations,
)
from utils.core.model_loader import ModelLoader


def main():
    model_type = "dqn"
    eval = True
    artifact_path = os.path.join("artifacts")

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact, agents=10)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=eval,
        artifact_path=artifact_path,
    )
    model = list(models.values())[-1]

    observations = collect_rollouts(
        env=environment,
        artifact=artifact,
        n=100,
        method="policy",
        force_update=True,
    )
    observations = filter_observations(observations)

    np.random.shuffle(observations)
    normalized_observations = normalize_observations(observations)
    data = normalized_observations[..., Observation.OBSERVATION][0:100]
    obs = observations_seperate_to_torch([d[0] for d in data])
    explainer = shap.GradientExplainer(model, obs)
    shap_values = explainer.shap_values(obs)

    mean_shap_values = np.array(shap_values[0]).mean(axis=-1)
    pixel_values = np.array(obs[0], copy=True)

    binary_to_rgb = {
        0: [255, 0, 0],  # Red
        1: [0, 255, 0],  # Green
        2: [0, 0, 255],  # Blue
        3: [255, 255, 0],  # Yellow
        4: [0, 255, 255],  # Cyan
    }
    rgb_images = []
    for pixel_value in pixel_values:
        image = []
        for row in pixel_value:
            image_row = []
            for cell in row:
                image_row.append(binary_to_rgb[np.argmax(cell)])
            image.append(image_row)
        rgb_images.append(image)
    rgb_images = np.array(rgb_images)

    for i in range(mean_shap_values.shape[0]):
        # NOTE: If the pixel values are not shown, its because image plot does something witih kmeans because the shape != 3 and the values is 0. Just go in the code and add 1e-10 so it doesn't devide by 0.
        shap.image_plot(mean_shap_values[i], rgb_images[i])


if __name__ == "__main__":
    main()
