from multigrid.envs.cleanup import CleanUpEnv
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
    agents = 1
    env = CleanUpEnv(boxes=2, agents=agents, width=11, height=11)
    controller = Controller(agents)
    while True:
        env.reset()
        run_episode(controller, env)
