from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from multiworld.utils.wrappers import ObservationCollectorWrapper
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType

env = GoToGoalEnv(
    width=10,
    height=10,
    max_steps=20,
    agents=1,
    preprocessing=PreprocessingEnum.ohe_minimal,
    success_termination_mode="all",
    render_mode="rgb_array",
)

env_wrapped = ObservationCollectorWrapper(env, observations=1000, sample_rate=1)

config = (
    DQNConfig(
        eps_start=0.0,
        eps_end=0.00,
    )
    .environment(env=env_wrapped)
    .network(network_type=NetworkType.MULTI_INPUT)
    .model("model_1000:v0")
    .debugging(log_level="INFO")
    .rendering()
)

dqn = DQN(config)

while True:
    dqn.learn()
