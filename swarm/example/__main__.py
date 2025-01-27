from multiworld.envs.flock import FlockEnv
from multiworld.base import MultiWorldEnv
from multiworld.example.controller import Controller


def run_episode(controller: Controller, env: MultiWorldEnv):
    while True:
        actions = controller.get_actions()
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        env.render()

        if all(terminations.values()) or all(truncations.values()):
            break


if __name__ == "__main__":
    agents = 3
    env = FlockEnv(
        width=100,
        height=100,
        max_steps=250,
        agents=agents,
        success_termination_mode="all",
        object_size=16,
        screen_size=(1000, 1000),
    )
    controller = Controller(agents, same_keys=True)
    while True:
        env.reset()
        run_episode(controller, env)
