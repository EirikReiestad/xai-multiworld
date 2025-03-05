import argparse
import logging

from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from utils.core.model_downloader import ModelDownloader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dms",
        "--download-models",
        nargs=5,
        metavar=("project_folder", "model_name", "low", "high", "step"),
        help="Download models with arguments: [project_folder] [low] [high] [step]",
    )
    parser.add_argument(
        "-dm",
        "--download-model",
        nargs="*",
        metavar=("model_name"),
        help="Download model with optional arguments: [project_folder] [model_name]",
    )
    args = parser.parse_args()

    model_name = "model"
    version = "latest"

    if args.download_models:
        project_folder, model_name, low, high, step = args.download_models
        low, high, step = int(low), int(high), int(step)
        models = [f"{model_name}_{i}:{version}" for i in range(low, high, step)]
    elif args.download_model:
        project_folder, model_name = args.download_model
        models = [str(model_name)]
    else:
        logger.info("No valid arguments provided.")
        return

    download(project_folder, model_name, models)


def download(project_folder: str, model_name: str, models: list[str]):
    env = GoToGoalEnv(
        goals=1,
        width=5,
        height=5,
        max_steps=100,
        preprocessing=PreprocessingEnum.ohe_minimal,
        agents=1,
        agent_view_size=7,
        success_termination_mode="all",
        render_mode="rgb_array",
    )

    dqn_config = (
        DQNConfig()
        .network(network_type=NetworkType.MULTI_INPUT)
        .environment(env=env)
        .training()
        .debugging(log_level="INFO")
        .rendering()
    )
    dqn = DQN(dqn_config)

    model_downloader = ModelDownloader(
        project_folder=project_folder,
        model_name=model_name,
        models=models,
        model=dqn.model,
    )
    model_downloader.download()


if __name__ == "__main__":
    main()
