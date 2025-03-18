import argparse
import glob
import json
import logging
import os
from copy import Error

import numpy as np
from PIL import Image

from multiworld.multigrid.core.action import Action
from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from utils.common.numpy_collections import NumpyEncoder
from utils.common.observation import (
    Observation,
    observation_data_to_numpy,
    observation_from_file,
    observation_from_observation_file,
    observation_to_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(path: str):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ro",
        "--render-observation",
        nargs="*",
        metavar=("filename"),
        help="Render the observations stored in json files under assets/concepts with optional arguement [filename]",
    )

    args = parser.parse_args()

    filename = None
    if args.render_observation:
        filename = args.render_observation[0]
        if not filename.endswith(".json"):
            filename += ".json"

    save_directory = os.path.join("assets", "rendered")

    if filename is not None:
        render(path, filename, save_directory, show=True)
        return

    matching_paths = glob.glob(os.path.join(path, "[0-9]*.json"))
    filenames = [os.path.basename(filepath) for filepath in matching_paths]
    for filename in filenames:
        render(path, filename, save_directory, show=False)


def render(directory: str, filename: str, save_directory: str, show: bool = False):
    path = os.path.join(directory, filename)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # observation: Observation = observation_from_file(path)
    observation = observation_from_observation_file(path)

    numpy_obs = observation_data_to_numpy(observation)
    grid = numpy_obs[0][0]
    width, height = grid.shape[:2]

    env = GoToGoalEnv(
        agents=1,
        width=width,
        height=height,
        preprocessing=PreprocessingEnum.ohe_minimal,
        render_mode="rgb_array",
    )
    env.reset()

    # np.random.shuffle(numpy_obs)

    save_filename = filename.replace(".json", ".png")

    save_mask = np.zeros(observation.shape[0], dtype=bool)
    observation_save_path = os.path.join(
        "assets", "custom", "multi_gtg_observations.json"
    )
    count = 0
    images = []
    for i, obs in enumerate(numpy_obs):
        env.update_from_numpy(obs)
        action = Action(observation[..., Observation.LABEL][i])
        logging.info(f"Action: {action, action.name}")
        while True:
            img = env.render()
            if img is not None:
                images.append(img)
                image = Image.fromarray(img, "RGB")
                path = os.path.join(save_directory, f"{count}_{save_filename}")
                image.save(path)
                if show:
                    image.show()
            if not show:
                break
            if input() == "s":
                save_mask[i] = True
                data = [
                    obs[0]
                    for obs in observation[..., Observation.OBSERVATION][save_mask]
                ]
                observation_to_file(data, observation_save_path)
                break
            if input() == "":
                break
        count += 1

    try:
        with open(os.path.join(directory, "similarity_weights.json"), "r") as f:
            similarity_weights = json.load(f)
        for key, values in reversed(similarity_weights.items()):
            if (
                filename.startswith(key)
                and len(values) == len(images)
                and "positive" in filename
            ):
                images = [image * (1 - weight) for image, weight in zip(images, values)]
                break
    except FileNotFoundError as e:
        pass

    if len(images) > 0:
        images = np.stack(images)
        image = np.mean(images, axis=0).astype(np.uint8)
        image = Image.fromarray(image, "RGB")
        path = os.path.join(save_directory, f"composite_{save_filename}")
        image.save(path)


if __name__ == "__main__":
    path = os.path.join("archive", "multi-gtg-15-20", "results")
    path = os.path.join("archive", "20250313-120443", "results")
    path = os.path.join("assets", "results")
    path = os.path.join("assets", "concepts")
    path = os.path.join("assets", "observations")

    main(path)
