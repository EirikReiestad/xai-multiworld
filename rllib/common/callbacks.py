from typing import Callable, Dict

from numpy.typing import NDArray

from multigrid.utils.typing import ObsType

RenderingCallback = Callable[[NDArray, ObsType], NDArray]


def empty_rendering_callback(
    image: NDArray, observations: Dict[str, ObsType]
) -> NDArray:
    return image
