from multigrid.envs.empty import EmptyEnv
from multigrid.base import MultiGridEnv
from multigrid.example.controller import Controller


def run_episode(agents: int, env: MultiGridEnv):
    while True:
        controller = Controller(agents)

        actions = controller.get_actions()
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        env.render()

        if all(terminations.values()) or all(truncations.values()):
            break


if __name__ == "__main__":
    agents = 1
    env = EmptyEnv(agents, width=11, height=11)
    env.reset()
    while True:
        run_episode(agents, env)
