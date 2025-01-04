from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List
from itertools import chain

from utils.common.observation import Observation


def image_to_element_matrix(
    image: NDArray, observation: NDArray, absolute: bool = False
) -> Dict[List, NDArray]:
    """
    Take in a image of values, and stores the corresponding value to the observation element.
    """
    assert (
        image.shape == observation.shape
    ), "Image and observation must have the same size, not {} and {}".format(
        image.shape, observation.shape
    )

    width, height = image.shape[:2]
    element_matrices = defaultdict(lambda: np.empty((width, height), dtype=object))

    for y in range(height):
        for x in range(width):
            i = (y, x)
            hashable_elem = tuple(observation[i])
            hashable_elem = (hashable_elem[0], 0, 0)
            element_matrices[hashable_elem][i] = image[i]
            if absolute:
                element_matrices[hashable_elem][i] = abs(image[i])

    return element_matrices


def images_to_element_matrix(
    images: NDArray,
    observations: Observation,
    average: bool = True,
    absolute: bool = False,
) -> Dict[List, NDArray]:
    width, height, channels = images[0].shape
    element_matrices = defaultdict(lambda: np.empty((width, height), dtype=object))

    def update_dict(elems: Dict[List, NDArray]):
        for key, value in elems.items():
            for i, value in np.ndenumerate(value):
                if value is None:
                    continue
                if element_matrices[key][i] is None:
                    element_matrices[key][i] = list()
                element_matrices[key][i].append(value)

    for image, observation in zip(images, observations):
        obs = np.array(observation.features[0]["image"])
        elems = image_to_element_matrix(image, obs, absolute)
        update_dict(elems)

    def element_matrix_average(
        element_matrices: Dict[List, NDArray],
    ) -> Dict[List, NDArray]:
        averaged_element_matrices = defaultdict(
            lambda: np.full((width, height, channels), np.nan, dtype=np.float32)
        )
        for key in element_matrices:
            for i, value in np.ndenumerate(element_matrices[key]):
                if value is None:
                    continue
                averaged_element_matrices[key][i] = np.mean(
                    value, axis=0, dtype=np.float32
                )

        return averaged_element_matrices

    if average:
        return element_matrix_average(element_matrices)
    return element_matrices


def flatten_element_matrices(matrices: Dict[List, NDArray]) -> Dict[List, NDArray]:
    return {key: flatten_element_matrix(value) for key, value in matrices.items()}


def flatten_element_matrix(matrix: NDArray) -> NDArray:
    flat_list = []
    for row in matrix:
        for col in row:
            if col is None:
                flat_list.append(np.nan)
                continue
            flat_list.extend(col)
    return np.array(flat_list, dtype=object)
