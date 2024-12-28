from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.algorithms.dqn.dqn import DQN
from multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.wrappers.dqn_concept_bottleneck_wrapper import DQNConceptBottleneckWrapper
from rllib.common.callbacks import RenderingCallback
from PIL import Image
import time
from xailib.core.concept_backpropagation.concept_backpropagation import (
    ConceptBackpropagation,
)

env = GoToGoalEnv(
    width=10,
    height=10,
    max_steps=100,
    agents=10,
    success_termination_mode="all",
    render_mode="human",
)


def rendering_callback(image, observations):
    obs = next(iter(observations.values()))

    feature_influence = ConceptBackpropagation().compute_feature_influence(obs, "goal")

    img_pil = Image.fromarray(image)
    img_pil.show()
    time.sleep(5)
    return image


config = (
    DQNConfig(
        batch_size=32,
        replay_buffer_size=10000,
        gamma=0.99,
        learning_rate=1e-4,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=100000,
        target_update=1000,
    )
    .environment(env=env)
    .training()
    .debugging(log_level="INFO")
    .rendering(callback=rendering_callback)
    # .wandb(project="multigrid-go-to-goal-cb")
)

dqn = DQN(config)
dqn = DQNConceptBottleneckWrapper(dqn)

while True:
    dqn.learn()
