from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from utils.core.model_downloader import ModelDownloader

project_folder = "go-to-goalv0"
model_name = "model"

models = []
for i in range(0, 1001, 200):
    models.append(f"model_{i}:v0")

env = GoToGoalEnv(render_mode="rgb_array")

dqn_config = (
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
