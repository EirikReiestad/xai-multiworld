from multigrid.envs.tag import TagEnv
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
    agents = 5
    env = TagEnv(
        width=10,
        height=10,
        max_steps=250,
        agents=agents,
        success_termination_mode="all",
    )
    controller = Controller(agents, same_keys=True)
    while True:
        env.reset()
        run_episode(controller, env)
