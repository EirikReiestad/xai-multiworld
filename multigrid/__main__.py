from multigrid.envs.empty import EmptyEnv
from itertools import count
import logging

logging.basicConfig(level=logging.INFO)

env = EmptyEnv(10)

for e in count():
    logging.info(f"Episode: {e}")
    env.reset()
    for t in count():
        logging.info(f"step: {t}")
        actions = env.action_space.sample()
        obs, rewards, terms, truncs, infos = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            break
        env.render()
