from multigrid.envs.boxwar import BoxWarEnv
from multigrid.base import MultiGridEnv
from multigrid.example.controller import Controller


def run_episode(controller: Controller, env: MultiGridEnv):
    while True:
        actions = controller.get_actions()
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        env.render()

        if all(terminations.values()) or all(truncations.values()):
            break


if __name__ == "__main__":
    agents = 2
    env = BoxWarEnv(boxes=2, agents=agents, width=11, height=11)
    controller = Controller(agents, same_keys=True)
    while True:
        env.reset()
        run_episode(controller, env)
