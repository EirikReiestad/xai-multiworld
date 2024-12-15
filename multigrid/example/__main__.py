from multigrid.envs.empty import EmptyEnv
from multigrid.example.controller import Controller

controller = Controller()


def run_episode(agents: int = 1):
    while True:
        action = controller.get_action(agents)
        next_observations, rewards, terminations, truncations, infos = env.step(actions)


if __name__ == "__main__":
    env = EmptyEnv()
    env.reset()
    while True:
        run_episode()
    env.render()
    env.render()
