import argparse
import logging
from typing import Literal

from multiworld.multigrid.base import MultiGridEnv
from multiworld.multigrid.utils.wrappers import MultiGridConceptObsWrapper
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from utils.common.environment import create_environment
from utils.core.model_loader import ModelLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gc",
        "--generate-concepts",
        nargs="*",
        metavar=("n"),
        help="Generate n concepts dataset (for all possible concepts)",
    )
    args = parser.parse_args()

    observations = 100
    method = "policy"
    if args.generate_concepts is not None and len(args.generate_concepts) > 0:
        observations = int(args.generate_concepts[0])
        method = (
            str(args.generate_concepts[1])
            if len(args.generate_concepts) > 1
            else "policy"
        )
        if method not in ["random", "policy"]:
            raise ValueError("Method must be either 'random' or 'policy'")

    artifact = ModelLoader.load_latest_model_artifacts_from_path()
    environment = create_environment(artifact)

    generate_concepts(observations, environment, method)


def generate_concepts(
    observations: int, env: MultiGridEnv, method: Literal["random", "policy"] = "policy"
):
    concepts = [
        "random",
        # "goal_in_front",
        # "goal_in_view",
        # "goal_to_left",
        # "goal_to_right",
        # "wall_in_view",
        # "agent_in_view",
        # "agent_to_right",
        # "agent_to_left",
        # "agent_in_front",
    ]

    logging.info(
        f"Generating {observations} concepts for {concepts} with {method} method"
    )

    env_wrapped = MultiGridConceptObsWrapper(
        env, observations=observations, concepts=concepts, method="random"
    )

    config = (
        DQNConfig(
            learning_rate=3e-4,
            eps_start=1.0 if method == "random" else 0.05,
            eps_end=1.0 if method == "random" else 0.05,
            update_method="soft",
            target_update=1000,
        )
        .network(network_type=NetworkType.MULTI_INPUT)
        .environment(env=env_wrapped)
        .debugging(log_level="INFO")
        .rendering()
    )
    dqn = DQN(config)

    while True:
        dqn.learn()


if __name__ == "__main__":
    main()
