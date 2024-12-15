from threading import Thread, Event

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
        self._action_event = Event()

        def listen():
            with Listener(
                on_press=self.on_press, on_release=self.on_release
            ) as listener:
                listener.join()

        # Start listener in a separate thread
        listen_thread = Thread(target=listen, daemon=True)
        listen_thread.start()

    def get_actions(self) -> dict[AgentID, int]:
        self._reset()
        self.wait_for_keypress()
        assert all(
            action is not None for action in self._actions
        ), "Not all agents have an action"
        return {str(i): self._actions[i] for i in range(self._agents)}

    def _reset(self):
        self._actions = [None] * self._agents
        self._current_action = None
        self._action_event.clear

    def on_press(self, key):
        try:
            if key.char in self.key_map:
                self._current_action = self.key_map[key.char]
                self._action_event.set()
        except AttributeError:
            pass

    def on_release(self, key):
        if key == "esc":
            return False

    def wait_for_keypress(self) -> int:
        while None in self._actions:
            self._action_event.wait()
            if self._current_action is None:
                continue
            for i in range(self._agents):
                if self._actions[i] is None:
                    self._actions[i] = self._current_action
                    break
                self._current_action = None
                self._action_event.clear()
