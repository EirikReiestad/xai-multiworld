"""
Uses Stable-Baselines3 to train agents to play the MPE environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""

from __future__ import annotations

import glob
import os
import time
from typing import Literal

import supersuit as ss
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

import wandb
from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.pettingzoo import PettingZooWrapper, to_pettingzoo_env
from wandb.integration.sb3 import WandbCallback


def train(
    env_fn,
    experiment_name: str,
    steps: int = 10_000,
    seed: int | None = 0,
    render_mode: Literal["human", "rgb_array"] | None = "human",
    render_interval: int = 5000,
    render_every_n_episodes: int = 10,
    **env_kwargs,
):
    env = env_fn(**env_kwargs)
    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    # Note: MPE's observation space is discrete, so we use an MLP policy rather than CNN
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256,
        tensorboard_log=f"runs/{experiment_name}",
    )

    total_steps = 0
    episode_count = 0
    while total_steps < steps:
        model.learn(
            total_timesteps=render_interval,
            reset_num_timesteps=False,
            # callback=WandbCallback(),
        )
        total_steps += render_interval
        episode_count += 1

        if episode_count % render_every_n_episodes == 0 and render_mode == "human":
            # Evaluation for rendering
            eval(env_fn, num_games=1, render_mode="human", **env_kwargs)

    model.save(
        f"models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    )

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    # env = ss.pad_observations_v0(env)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using the same model for every agent
    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                if obs is None:
                    act = env.action_space(agent).sample()
                else:
                    obs = obs.reshape(
                        (1, -1)
                    )  # Ensure the observation has the correct shape
                    act, _ = model.predict(obs, deterministic=True)
                    act = (
                        act
                        if env.action_space(agent).contains(act)
                        else env.action_space(agent).sample()
                    )  # Ensure action is in action space

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    env_fn = to_pettingzoo_env(GoToGoalEnv)

    env_kwargs = {}

    config = {"policy_type": "MultiInputPolicy", "total_timesteps": 25000}
    experiment_name = f"mpe_{int(time.time())}"
    """
    wandb.init(
        name=experiment_name,
        project="test",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    """

    # Train a model (takes ~3 minutes on GPU)
    train(env_fn, experiment_name=experiment_name, steps=100_000, seed=0, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 10 games
    eval(env_fn, num_games=10, render_mode="human", **env_kwargs)
