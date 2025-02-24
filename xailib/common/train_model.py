import logging
import gc
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def train_model(
    model: nn.Module,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    dataset: TensorDataset,
    test_split: float,
    val_split: float,
    verbose: bool = False,
) -> Tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()

    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    if verbose:
        logging.info(f"Training model with {len(train_dataset)} samples")
        logging.info(f"Validating model with {len(val_dataset)} samples")
        logging.info(f"Testing model with {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

        if verbose:
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

    if verbose:
        logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    del model
    gc.collect()

    return test_loss, test_accuracy
