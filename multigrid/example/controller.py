from threading import Thread

from pynput.keyboard import Listener

from multigrid.core.action import Action
from multigrid.utils.typing import AgentID


class Controller:
    def __init__(self, agents: int = 1):
        self._agents = agents

        self.key_map = {
            "a": Action.left,  # Turn left
            "d": Action.right,  # Turn right
            "w": Action.forward,  # Move forward
            "p": Action.pickup,  # Pick up object
            "o": Action.drop,  # Drop object
            "t": Action.toggle,  # Toggle object
            "e": Action.done,  # Done task
        }
        self._current_action = None
        self._actions = [None] * agents

    def get_actions(self) -> dict[AgentID, int]:
        self.wait_for_keypress()
        assert all(
            action is not None for action in self._actions
        ), "Not all agents have an action"
        return {str(i): self._actions[i] for i in range(self._agents)}

    def on_press(self, key):
        try:
            if key.char in self.key_map:
                self._current_action = self.key_map[key.char]
        except AttributeError:
            pass

    def on_release(self, key):
        if key == "esc":
            return False

    def wait_for_keypress(self) -> int:
        def listen():
            with Listener(
                on_press=self.on_press, on_release=self.on_release
            ) as listener:
                listener.join()

        # Start listener in a separate thread
        listen_thread = Thread(target=listen)
        listen_thread.daemon = True  # Exit thread when the main program ends
        listen_thread.start()

        while self._current_action is None:
            # Keep looping until we have a valid action
            pass

        print(self._current_action)

        for i in range(self._agents):
            if self._actions[i] is None:
                self._actions[i] = self._current_action
                break

        self._current_action = None
