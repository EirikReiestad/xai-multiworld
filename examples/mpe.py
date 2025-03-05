"""
Uses Stable-Baselines3 to train agents to play the MPE environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""

from __future__ import annotations

import glob
import logging
import os
import time

import supersuit as ss
from pettingzoo.mpe import simple_tag_v3
from stable_baselines3 import PPO

import wandb
from wandb.integration.sb3 import WandbCallback

logging.basicConfig(level=logging.INFO)


def train_mpe_supersuit(
    env_fn,
    steps: int = 10_000,
    seed: int | None = 0,
    **env_kwargs,
):
    experiment_name = f"mpe_{int(time.time())}"
    run = wandb.init(
        name=experiment_name,
        project="mpe",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    env = env_fn.parallel_env(**env_kwargs)
    env = ss.pad_observations_v0(env)

    env.reset(seed=seed)

    logging.info(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")

    model.learn(
        total_timesteps=steps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    model.save(
        f"models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    )

    logging.info("Model has been saved.")

    logging.info(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = ss.pad_observations_v0(env)

    logging.info(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        logging.warning("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

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
                    obs = obs.reshape((1, -1))
                    act, _ = model.predict(obs, deterministic=True)
                    act = (
                        act
                        if env.action_space(agent).contains(act)
                    
                    )

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    logging.info("Rewards: ", rewards)
    logging.info(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    env_fn = simple_tag_v3
    env_kwargs = {}

    train_mpe_supersuit(
        env_fn,
        steps=1_000_000,
        seed=0,
        **env_kwargs,
        render_mode="rgb_array",
    )

    eval(env_fn, num_games=10, render_mode="human", **env_kwargs)
