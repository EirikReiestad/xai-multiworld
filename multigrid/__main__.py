from multigrid.envs.empty import EmptyEnv
from itertools import count
import logging

logging.basicConfig(level=logging.INFO)

env = EmptyEnv(10)

for t in count():
    logging.info(f"step: {t}")
    env.reset()
    actions = env.action_space.sample()
    env.step(actions)
    env.render()
