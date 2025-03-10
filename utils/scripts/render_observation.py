import argparse
import logging
import os

import numpy as np

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

    path = os.path.join("assets", "observations", filename)
    path = os.path.join("pipeline", "20250310-160946", "results", filename)

    if not path.endswith(".json"):
        path += ".json"

    render(path)


def render(filename: str):
    observation: Observation = observation_from_file(filename)
    numpy_obs = observation_data_to_numpy(observation)
    grid = numpy_obs[0][0]
    width, height = grid.shape[:2]

    env = GoToGoalEnv(
        agents=1,
        width=width,
        height=height,
        preprocessing=PreprocessingEnum.ohe_minimal,
        render_mode="human",
    )
    env.reset()

    np.random.shuffle(numpy_obs)

    for obs in numpy_obs:
        env.update_from_numpy(obs)
        while True:
            env.render()
            if input() == "":
                break


if __name__ == "__main__":
    main()
