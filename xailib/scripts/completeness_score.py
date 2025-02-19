import logging
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset, random_split

from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from rllib.algorithms.dqn.dqn import DQN
from rllib.algorithms.dqn.dqn_config import DQNConfig
from rllib.core.network.network import NetworkType
from utils.common.observation import (
    Observation,
    load_and_split_observation,
    observations_from_file,
    zip_observation_data,
)
from utils.core.model_loader import ModelLoader
from xailib.common.activations import (
    compute_activations_from_models,
)
from xailib.common.binary_concept_score import individual_binary_concept_score
from xailib.common.probes import get_probes

logging.basicConfig(level=logging.INFO)


class SimpleNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


def run(concepts: List[str], dqn: DQN):
    probes = {}
    concept_scores = []
    layer_idx = 2
    for i, concept in enumerate(concepts):
        probe = get_probe(concept, layer_idx)
        probes[i] = probe

    observations = observations_from_file(
        os.path.join("assets/observations", "observations" + ".json")
    )

    concept_scores = np.array(
        get_concept_score(observations, probes, layer_idx), dtype=np.float32
    )
    labels = np.array(observations[..., Observation.LABEL], dtype=np.float32)

    model = SimpleNetwork(len(concepts), 500, int(dqn.action_space.n))
    train(model, concept_scores, labels)


def train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.2,
    test_split: float = 0.1,
):
    epochs = 10
    batch_size = 64
    learning_rate = 0.001

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()

    # Create dataset and split into training, validation, and test sets
    dataset = TensorDataset(torch.from_numpy(X), torch.tensor(y, dtype=torch.long))
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    logging.info(f"Training model with {len(train_dataset)} samples")
    logging.info(f"Validating model with {len(val_dataset)} samples")
    logging.info(f"Testing model with {len(test_dataset)} samples")

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        logging.info(
            f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, "
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

        model.train()

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


def get_concept_score(
    observation: Observation, probes: Dict[str, LogisticRegression], layer_idx: int
):
    ignore = []
    observation_zipped = zip_observation_data(observation)

    model = ModelLoader.load_latest_model_from_path("artifacts", dqn.model)
    models = {"latest": model}

    activations, input, output = compute_activations_from_models(
        models, observation_zipped, ignore
    )
    layer_activations = list(activations["latest"].values())[layer_idx]

    concept_scores = []
    for concept, probe in probes.items():
        concept_score = individual_binary_concept_score(layer_activations, probe)
        concept_scores.append(concept_score)

    concept_scores = list(zip(*concept_scores))
    return concept_scores


def get_probe(concept: str, layer_idx: int):
    ignore = []

    model = ModelLoader.load_latest_model_from_path("artifacts", dqn.model)
    models = {"latest": model}
    positive_observation, test_observation = load_and_split_observation(concept, 0.8)
    negative_observation, _ = load_and_split_observation("negative_" + concept, 0.8)

    test_observation_zipped = zip_observation_data(test_observation)

    test_activations, test_input, test_output = compute_activations_from_models(
        models, test_observation_zipped, ignore
    )

    probes = get_probes(models, positive_observation, negative_observation, ignore)

    layer_activations = list(test_activations["latest"].values())[layer_idx]
    probe = list(probes["latest"].values())[layer_idx]

    return probe


if __name__ == "__main__":
    import os

    # os.chdir("../../")
    artifact = ModelLoader.load_latest_model_artifacts_from_path("artifacts")

    width = artifact.metadata.get("width")
    height = artifact.metadata.get("height")
    agents = artifact.metadata.get("agents")
    conv_layers = artifact.metadata.get("conv_layers")
    hidden_units = artifact.metadata.get("hidden_units")
    eps_threshold = artifact.metadata.get("eps_threshold")
    learning_rate = artifact.metadata.get("learning_rate")

    env = GoToGoalEnv(
        width=10,
        height=10,
        max_steps=10,
        agents=1,
        preprocessing=PreprocessingEnum.ohe_minimal,
        success_termination_mode="all",
        render_mode="human",
    )

    config = (
        DQNConfig(
            batch_size=128,
            replay_buffer_size=10000,
            gamma=0.99,
            learning_rate=3e-4,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=50000,
            target_update=1000,
        )
        .network(
            network_type=NetworkType.MULTI_INPUT,
            conv_layers=conv_layers,
            hidden_units=hidden_units,
        )
        .debugging(log_level="INFO")
        .environment(env=env)
    )

    dqn = DQN(config)

    # concepts = concept_checks.keys()
    concepts = ["random"]
    concepts = [
        # "random",
        # "agent_in_front",
        # "agent_in_view",
        # "agent_to_left",
        # "agent_to_right",
        "goal_in_front",
        "goal_in_view",
        "goal_to_left",
        "goal_to_right",
        "wall_in_view",
    ]

    run(concepts, dqn)
