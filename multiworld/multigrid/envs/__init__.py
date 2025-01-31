"""
************
Environments
************

This package contains implementations of several MultiGrid environments.

**************
Configurations
**************

* `Blocked Unlock Pickup <./multigrid.envs.blockedunlockpickup>`_
    * ``MultiGrid-BlockedUnlockPickup-v0``
* `Empty <./multigrid.envs.empty>`_
    * ``MultiGrid-Empty-5x5-v0``
    * ``MultiGrid-Empty-Random-5x5-v0``
    * ``MultiGrid-Empty-6x6-v0``
    * ``MultiGrid-Empty-Random-6x6-v0``
    * ``MultiGrid-Empty-8x8-v0``
    * ``MultiGrid-Empty-16x16-v0``

from gymnasium.envs.registration import register

from .empty import EmptyEnv

CONFIGURATIONS = {
    "MultiGrid-Empty-5x5-v0": (EmptyEnv, {"width": 5, "height": 5}),
    "MultiGrid-Empty-Random-5x5-v0": (
        EmptyEnv,
        {"width": 5, "height": 5, "agent_start_pos": None},
    ),
    "MultiGrid-Empty-6x6-v0": (EmptyEnv, {"width": 6, "height": 6}),
    "MultiGrid-Empty-Random-6x6-v0": (
        EmptyEnv,
        {"width": 6, "height": 6, "agent_start_pos": None},
    ),
    "MultiGrid-Empty-8x8-v0": (EmptyEnv, {}),
    "MultiGrid-Empty-16x16-v0": (EmptyEnv, {"width": 16, "height": 16}),
}

# Register environments with gymnasium

for name, (env_cls, config) in CONFIGURATIONS.items():
    register(id=name, entry_point=env_cls, kwargs=config)
"""
