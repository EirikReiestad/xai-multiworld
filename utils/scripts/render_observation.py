import argparse
import logging
import os

import numpy as np
from PIL import Image

from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from utils.common.observation import (
    Observation,
    observation_data_to_numpy,
    observation_from_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ro",
        "--render-observation",
        nargs=1,
        metavar=("filename"),
        help="Render the observations stored in json files under assets/concepts with optional arguement [filename]",
    )

    args = parser.parse_args()

    if args.render_observation:
        filename = args.render_observation[0]
    else:
        logger.info("No valid arguments provided.")
        return

    path = os.path.join("pipeline", "20250310-160946", "results", filename)
    path = os.path.join("assets", "results")

    if not filename.endswith(".json"):
        filename += ".json"

    save_directory = os.path.join("assets", "rendered")
    render(path, filename, save_directory, show=False)


def render(directory: str, filename: str, save_directory: str, show: bool = False):
    path = os.path.join(directory, filename)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    observation: Observation = observation_from_file(path)
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

    np.random.shuffle(numpy_obs)

    save_filename = filename.replace(".json", ".png")

    count = 0
    for obs in numpy_obs:
        env.update_from_numpy(obs)
        while True:
            img = env.render()
            if img is not None:
                image = Image.fromarray(img, "RGB")
                path = os.path.join(save_directory, f"{count}_{save_filename}")
                image.save(path)
                if show:
                    image.show()
            if not show:
                break
            if input() == "":
                break
        count += 1


if __name__ == "__main__":
    main()
