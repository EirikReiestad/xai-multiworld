import logging
import os

import matplotlib.pyplot as plt
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

logging.basicConfig(level=logging.INFO)


def main():
    model_type = "dqn"
    eval = True
    artifact_path = os.path.join("artifacts")
    save_path = os.path.join("assets", "rendered")
    split = 0.8

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
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
        n=500,
        method="policy",
        force_update=True,
    )
    observations = filter_observations(observations)

    split_size = int(split * len(observations))

    np.random.shuffle(observations)
    normalized_observations = normalize_observations(observations)
    data = normalized_observations[..., Observation.OBSERVATION][:split_size]
    obs = observations_seperate_to_torch([d[0] for d in data])
    logging.info("Running SHAP explainer...")
    explainer = shap.GradientExplainer(model, obs)
    data = normalized_observations[..., Observation.OBSERVATION][split_size:]
    obs = observations_seperate_to_torch([d[0] for d in data])
    logging.info("Calculating SHAP values...")
    shap_values = explainer.shap_values(obs)

    mean_shap_values = np.array(shap_values[0]).mean(axis=-1)
    pixel_values = np.array(obs[0], copy=True)

    grid = pixel_values[0]
    width, height = grid.shape[:2]
    environment = create_environment(artifact, width=width, height=height)

    logging.info("Rendering images")
    rgb_images = []
    for pixel_value in pixel_values:
        data = np.array([pixel_value, pixel_value])
        environment.update_from_numpy(data)
        image = environment.render()
        rgb_images.append(image)
    rgb_images = np.array(rgb_images)

    for i in range(mean_shap_values.shape[0]):
        # NOTE: If the pixel values are not shown, its because image plot does something witih kmeans because the shape != 3 and the values is 0. Just go in the code and add 1e-10 so it doesn't devide by 0.
        shap.image_plot(mean_shap_values[i], rgb_images[i], show=False)
        plt.savefig(os.path.join(save_path, f"{i}_shap.png"))
        plt.close()


if __name__ == "__main__":
    main()
