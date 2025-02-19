from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from multiworld.multigrid.utils.wrappers import MultiGridConceptObsWrapper
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType

env = GoToGoalEnv(
    goals=1,
    static=True,
    width=10,
    height=10,
    max_steps=20,
    preprocessing=PreprocessingEnum.ohe_minimal,
    agents=1,
    agent_view_size=7,
    success_termination_mode="all",
    render_mode="human",
)

concepts = ["random"]
concepts = None
concepts = [
    "random",
    "random_negative",
    "goal_in_view",
    "goal_to_right",
    "goal_to_left",
    "goal_in_front",
    "wall_in_view",
    # "agent_in_view",
    # "agent_to_right",
    # "agent_to_left",
    # "agent_in_front",
]

env_wrapped = MultiGridConceptObsWrapper(
    env, observations=50, concepts=concepts, method="policy"
)

config = (
    DQNConfig(
        eps_start=0.0,
        eps_end=0.00,
    )
    .network(network_type=NetworkType.MULTI_INPUT)
    .environment(env=env_wrapped)
    .model("model_150:v0")
    .debugging(log_level="INFO")
    .rendering()
)

dqn = DQN(config)

while True:
    dqn.learn()
