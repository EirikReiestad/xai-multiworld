import os

import numpy as np

from utils.common.observation import Observation


class ObservationCollection:
    positive_observation: Observation
    negative_observation: Observation

    def __init__(
        self,
        positive_observation_path: str | None = None,
        negative_observation_path: str | None = None,
        folder_path: str = "artifacts/concepts",
    ) -> None:
        self._folder_path = folder_path

        if positive_observation_path is not None:
            self.load_positive_observations_from_path(positive_observation_path)
        if negative_observation_path is not None:
            self.load_negative_observations_from_path(negative_observation_path)

    def load_positive_observations_from_path(self, path: str) -> None:
        self.positive_observations = self._observation_from_file(path)

    def load_negative_observations_from_path(self, path: str) -> None:
        self.negative_observations = self._observation_from_file(path)

    def load_positive_observations(self, observations: list[Observation]) -> None:
        self.positive_observations = observations

    def load_negative_observations(self, observations: list[Observation]) -> None:
        self.negative_observations = observations

    def _observation_from_file(self, path: str) -> list[Observation]:
        self._check_file_exists(path)
        with open(path, "r") as f:
            observations = []
            for line in f:
                id, *data, label = line.strip().split(",")
                observations.append(Observation(id, np.array(data), label))
        return observations

    def _check_file_exists(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")

    def write(self, filename: str) -> None:
        positive_file_path = os.path.join(
            self._folder_path, "positive_observations", filename
        )
        negative_file_path = os.path.join(
            self._folder_path, "negative_observations", filename
        )
        self.write_data_to_file(positive_file_path)
        self.write_data_to_file(negative_file_path)

    def write_data_to_file(self, path: str) -> None:
        with open(path, "w") as f:
            for observation in self.positive_observations:
                f.write(
                    f"{observation.id},{','.join(map(str, observation.data))},{observation.label}\n"
                )
            for observation in self.negative_observations:
                f.write(
                    f"{observation.id},{','.join(map(str, observation.data))},{observation.label}\n"
                )


class ObservationCollectionManager:
    def __init__(self) -> None:
        self._data = ObservationCollection()

    def load_data_from_path(
        self, positive_observations_path: str, negative_observations_path: str
    ) -> None:
        self._data = ObservationCollection(
            positive_observations_path, negative_observations_path
        )

    def load_positive_observations_from_path(self, path: str) -> None:
        self._data.load_positive_observations_from_path(path)

    def load_negative_observations_from_path(self, path: str) -> None:
        self._data.load_negative_observations_from_path(path)

    def load_positive_observations(self, observations: list[Observation]) -> None:
        self._data.load_positive_observations(observations)

    def load_negative_observations(self, observations: list[Observation]) -> None:
        self._data.load_negative_observations(observations)

    def write(self, filename: str) -> None:
        self._data.write(filename)
