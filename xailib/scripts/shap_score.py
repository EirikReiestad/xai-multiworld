import json
import os

import numpy as np
import shap

from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.utils.wrappers import Observations
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.utils.torch.processing import observations_seperate_to_torch
from utils.common.observation import normalize_observations

if __name__ == "__main__":
    path = os.path.join("assets", "observations", "observations.json")
    if not os.path.exists(path):
        raise FileExistsError(f"Observation file does not exists at {path}")

    with open(path, "r") as f:
        data = json.load(f)

    observations = [Observations.deserialize(d) for d in data]

    agent_observations = [o.observations for o in observations]
    flattened_agent_observations = [
        o for obs in agent_observations for o in obs.values()
    ]

    env = GoToGoalEnv(render_mode="rgb_array")
    config = (
        DQNConfig(
            batch_size=16,
            replay_buffer_size=10000,
            gamma=0.99,
            learning_rate=1e-4,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=50000,
            target_update=1000,
        )
        .debugging(log_level="INFO")
        .environment(env=env)
    )

    dqn = DQN(config)

    model = dqn.model

    normalized_observations = normalize_observations(flattened_agent_observations)
    obs = observations_seperate_to_torch(flattened_agent_observations)
    explainer = shap.GradientExplainer(model, obs)
    shap_values = explainer.shap_values(obs)

    mean_shap_values = np.array(shap_values[0]).mean(axis=-1)

    pixel_values = np.array(obs[0])

    for idx in range(mean_shap_values.shape[0]):
        shap.image_plot(mean_shap_values[idx], pixel_values[idx])
