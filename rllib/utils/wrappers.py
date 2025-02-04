from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType


class GymnasiumWrapper(gym.Wrapper):
    @property
    def observation_space(self) -> spaces.Space[ObsType] | spaces.Space[WrapperObsType]:
        observation_space = super().observation_space
        return spaces.Dict(
            {
                0: spaces.Dict(
                    {
                        "observation": observation_space,
                        "other": spaces.Discrete(1),
                    }
                )
            }
        )

    @property
    def action_space(self) -> spaces.Space[ActType] | spaces.Space[WrapperActType]:
        action_space = super().action_space
        return spaces.Dict({0: action_space})

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, other = super().reset(seed=seed, options=options)
        return {0: {"observation": obs}}, other

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = [action.get(0)]
        observation, rewards, terminations, truncations, infos = super().step(action)
        return (
            {0: {"observation": observation}},
            {0: rewards},
            {0: terminations},
            {0: truncations},
            {0: infos},
        )
