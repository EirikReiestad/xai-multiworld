import argparse
import logging

from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.wrappers import MultiGridConceptObsWrapper
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType

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
    if args.generate_concepts is not None and len(args.generate_concepts) > 0:
        observations = int(args.generate_concepts[0])

    generate_concepts(observations)


def generate_concepts(observations: int):
    env = GoToGoalEnv(
        width=15,
        height=15,
        max_steps=200,
        agents=2,
        success_termination_mode="all",
        render_mode="human",
    )
    concepts = ["random"]
    concepts = None

    logging.info(f"Generating {observations} concepts for {concepts}")

    env_wrapped = MultiGridConceptObsWrapper(
        env, observations=observations, concepts=concepts, method="random"
    )

    config = (
        DQNConfig(
            batch_size=64,
            replay_buffer_size=10000,
            gamma=0.99,
            learning_rate=3e-4,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=100000,
            target_update=1000,
        )
        .network(network_type=NetworkType.MULTI_INPUT)
        .environment(env=env_wrapped)
        .training()
        .debugging(log_level="INFO")
        .rendering()
    )

    dqn = DQN(config)

    while True:
        dqn.learn()


if __name__ == "__main__":
    main()
