import gc
import logging
import os
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils.common.write import write_results
from xailib.utils.logging import (
    log_decision_tree_feature_importance,
)


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


def train_decision_tree(
    model: DecisionTreeClassifier,
    dataset: TensorDataset,
    test_split: float,
    feature_names: List[str],
    epochs: int = 20,
    result_path: str = os.path.join("assets", "results"),
    figure_path: str = os.path.join("assets", "figures"),
    show: bool = False,
    verbose: bool = False,
):
    results = []
    for _ in range(epochs):
        test_size = int(len(dataset) * test_split)
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        if verbose:
            logging.info(f"Training model with {len(train_dataset)} samples")
            logging.info(f"Testing model with {len(test_dataset)} samples")

        X_train = train_dataset[:][0].numpy()
        y_train = train_dataset[:][1].numpy()

        X_test = test_dataset[:][0].numpy()
        y_test = test_dataset[:][1].numpy()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if verbose:
            logging.info(f"Test set accuracy: {accuracy:.4f}")

        plt.figure(figsize=(20, 10))
        feature_names += ["random"]
        plot_tree(
            model,
            filled=True,
            feature_names=feature_names,  # class_names=data.target_names
        )
        if show:
            plt.show()
        plt.savefig(os.path.join(figure_path, "decision_tree.png"))

        feature_importances = model.feature_importances_
        feature_split_counts = np.zeros(len(feature_names))

        def count_feature_splits(node):
            if node == -1:
                return
            feature = model.tree_.feature[node]
            if feature != -2:
                feature_split_counts[feature] += 1
                count_feature_splits(model.tree_.children_left[node])
                count_feature_splits(model.tree_.children_right[node])

        count_feature_splits(0)

        result = {}
        for name, importance, count in zip(
            feature_names, feature_importances, feature_split_counts
        ):
            result[name] = [float(importance), float(count)]
        results.append(result)

    average_results = defaultdict(list[float])
    for key in results[0].keys():
        average_results[key] = [0, 0]

    for result in results:
        for key, value in result.items():
            importance, splits = value[0], value[1]
            average_results[key][0] += importance
            average_results[key][1] += splits
    for key in average_results.keys():
        average_results[key][0] /= epochs
        average_results[key][1] /= epochs

    path = os.path.join(result_path, "concept_score_decicion_tree.json")
    write_results(average_results, path)
    log_decision_tree_feature_importance(average_results)

    logging.info(f"\nTree Depth: {model.get_depth()}")
    logging.info(f"Number of Leaves: {model.get_n_leaves()}")

    path = os.path.join(result_path, "concept_score_decicion_tree_info.json")
    result = {
        "tree_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
    }

    write_results(result, path)
