from typing import Dict, List, Tuple

from utils.common.observation import Observation, load_and_split_observation


def get_observations(
    concepts: List[str],
) -> Tuple[
    Dict[str, Observation],
    Dict[str, Observation],
    Dict[str, Observation],
    Dict[str, Observation],
]:
    positive_observations = {}
    negative_observations = {}
    test_positive_observations = {}
    test_negative_observations = {}

    for concept in concepts:
        positive_observation, test_observation = load_and_split_observation(
            concept, 0.8
        )
        negative_observation, _ = load_and_split_observation("negative_" + concept, 0.8)

        positive_observations[concept] = positive_observation
        negative_observations[concept] = negative_observation
        test_positive_observations[concept] = test_observation
        test_negative_observations[concept] = negative_observation

    return (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    )
