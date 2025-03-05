"""Uses Stable-Baselines3 to train agents in the Knights-Archers-Zombies environment using SuperSuit vector envs.

This environment requires using SuperSuit's Black Death wrapper, to handle agent death.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""

from __future__ import annotations

import glob
import logging
import os
import time

import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

import wandb
from wandb.integration.sb3 import WandbCallback

logging.basicConfig(level=logging.INFO)


def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    experiment_name = f"mpe_{int(time.time())}"
    run = wandb.init(
        name=experiment_name,
        project="kaz",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    env = env_fn.parallel_env(**env_kwargs)

    env = ss.black_death_v3(env)

    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    logging.info(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    model = PPO(
        CnnPolicy if visual_observation else MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
    )

    model.learn(
        total_timesteps=steps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    logging.info("Model has been saved.")

    logging.info(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    logging.info(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        logging.info("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            if termination or truncation:
                act = None
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]

            env.step(act)

            for a in env.agents:
                rewards[a] += env.rewards[a]
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    logging.info(f"Avg reward: {avg_reward}")
    logging.info("Avg reward per agent, per game: ", avg_reward_per_agent)
    logging.info("Full rewards: ", rewards)
    return avg_reward


if __name__ == "__main__":
    env_fn = knights_archers_zombies_v10

    # Set vector_state to false in order to use visual observations (significantly longer training time)
    env_kwargs = dict(max_cycles=100, max_zombies=4, vector_state=True)

    # Train a model (takes ~5 minutes on a laptop CPU)
    train(env_fn, steps=81_920, seed=0, **env_kwargs)

    # Evaluate 10 games (takes ~10 seconds on a laptop CPU)
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 2 games (takes ~10 seconds on a laptop CPU)
    eval(env_fn, num_games=2, render_mode="rgb_array", **env_kwargs)
