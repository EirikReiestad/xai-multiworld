from multigrid.envs.take_box import TakeBox
from itertools import count
import logging

logging.basicConfig(level=logging.INFO)

env = TakeBox(5, width=10, height=10)

for e in count():
    logging.info(f"Episode: {e}")
    env.reset()
    for t in count():
        # logging.info(f"step: {t}")
        actions = env.action_space.sample()
        obs, rewards, terms, truncs, infos = env.step(actions)
        import time

        # time.sleep(5)
        if all(terms.values()) or all(truncs.values()):
            break
        env.render()
