import json
import random
import itertools
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from numpy.random import shuffle
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split

from experiments.src.compute_statistics import calculate_shapley_values
from experiments.src.model_handler import test_model, train_model
from experiments.src.network import FFNet
from experiments.src.utils import get_combinations


def get_neural_network_feature_importance(
    M: int,
    X: np.ndarray,
    y,
    output_size,
    log_interval,
    dry_run,
    gamma,
    batch_size,
    test_batch_size,
    iteration: int,
    epochs: int = 10,
    result_path: str = "experiments/results",
    filename: str = "completeness_score_network.json",
):
    y = torch.Tensor(y)
    X = torch.Tensor(X)
    print(f"Input shape: {X.shape}, target shape{y.shape}")
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print(f"Training set size: {train_size}, test set size: {test_size}")

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    lr = 0.1
    epochs = 20

    model = FFNet(M, output_size)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    sub_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    sub_test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    patience = 5
    patience_count = 0

    best_accuracy = 0
    train_acc = 0
    test_acc = 0

    for epoch in range(epochs):
        train_acc = train_model(
            log_interval,
            dry_run,
            model,
            None,
            sub_train_loader,
            optimizer,
            epoch,
        )
        test_acc = test_model(model, None, sub_test_loader)

        scheduler.step()
        patience_count += 1
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            patience_count = 0

        if patience_count == patience:
            break
        break

    results = defaultdict()
    best_test_acc = 0
    combs = get_combinations(list(range(M)))
    shuffle(combs)
    max_combs = 200
    n_subsets = 5
    combinations = []
    for comb in combs[:max_combs]:
        for _ in range(n_subsets):
            subset_size = random.randint(1, len(comb))
            subset = sorted(tuple(random.sample(comb, subset_size)))
            if subset in combinations:
                continue
            combinations.append(subset)
    for i, comb in enumerate(combinations):
        sub_X = create_sub_X(train_dataset, test_dataset, comb)
        test_targets = torch.stack(
            [test_dataset.dataset[i][1] for i in test_dataset.indices]
        )
        sub_dataset = torch.utils.data.TensorDataset(sub_X, test_targets)
        sub_test_loader = torch.utils.data.DataLoader(
            sub_dataset, batch_size=test_batch_size, shuffle=False
        )
        test_acc, test_loss = test_model(model, None, sub_test_loader)
        print(
            f"\n\n===== Computing accuracy for {comb} ({i}/{max_combs}) accuracy: {test_acc} ====="
        )
        best_test_acc = max(best_test_acc, test_acc)
        results[tuple(sorted(comb))] = (test_acc, test_loss)

    path = os.path.join(result_path, filename)
    with open(path, "w") as f:
        json_results = {str(key): value for key, value in results.items()}
        json.dump(json_results, f, indent=4)
    shapley_values = calculate_shapley_values(results, list(range(M)))
    if shapley_values == {}:
        return None, None
    shapley_values_results = {}
    for key, value in sorted(shapley_values.items()):
        shapley_values_results[key] = [value, 0]

    path = os.path.join(result_path, f"shapley_{iteration}_{filename}")
    with open(path, "w") as f:
        json.dump(shapley_values, f, indent=4)
    path = os.path.join(result_path, f"nn_feature_importances_{iteration}.json")
    with open(path, "w") as f:
        json.dump(shapley_values_results, f, indent=4)

    return best_test_acc, shapley_values


def get_neural_network_completeness_score(
    M,
    concept_scores_train,
    all_train_targets,
    output_size,
    batch_size,
    test_batch_size,
    log_interval,
    dry_run,
    gamma,
    iteration,
    result_path,
):
    print("\nBaseline test with Neural Network trained on concepts")
    print(concept_scores_train.shape)
    model = FFNet(M, output_size)
    concept_scores_train = torch.Tensor(concept_scores_train)
    all_train_targets = torch.Tensor(all_train_targets)
    print(
        f"Input shape: {concept_scores_train.shape}, target shape{all_train_targets.shape}"
    )
    dataset = TensorDataset(concept_scores_train, all_train_targets)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print(f"Training set size: {train_size}, test set size: {test_size}")

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    stats_file = f"{result_path}/nn_concept_training_stats_{iteration}.txt"
    all_stats = []
    lr = 0.1
    epochs = 20
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    best_test_acc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_acc = train_model(
            log_interval,
            dry_run,
            model,
            None,
            train_loader,
            optimizer,
            epoch,
        )
        test_acc, test_loss = test_model(model, None, test_loader)
        scheduler.step()
        epoch_time = time.time() - start_time
        stats = {
            "epoch": epoch,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "epoch_time": epoch_time,
        }
        all_stats.append(stats)
        print(
            f"Epoch {epoch} | Train Acc: {train_acc} | Test Acc: {test_acc} | Time: {epoch_time} sec"
        )
        # Early stopping logic
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=4)
    return best_test_acc, {}


def calculate_weights(model, cavs, layer_name, data) -> dict:
    # NOTE: This is for CNN
    weights = {}
    for i, (key, cav_data) in enumerate(data.items()):
        acts = get_activations(model, cav_data, layer_name)
        acts_flatten = acts.view(acts.size(0), -1)
        weight = acts_flatten * cavs[i]
        weight_original = weight.view(weight.size(0), *acts.shape[1:])
        weight_original = weight_original.sum(dim=1)

        size = cav_data[0].shape[1:]
        x_resized = np.zeros((weight_original.shape[0], *size))
        for i in range(weight_original.shape[0]):
            numpy_x = weight_original[i].detach().numpy()
            x_resized[i] = cv2.resize(
                numpy_x,
                size,
                interpolation=cv2.INTER_LINEAR,
            )
        weights[key] = x_resized
    return weights


def get_activations(model, data, layer_name):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    hook = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break

    if hook is None:
        raise ValueError(
            f"{layer_name} is not a valid layer. Possible layers: {[name for name, _ in model.named_modules()]}"
        )

    model.eval()
    with torch.no_grad():
        # data = data.to(next(model.parameters()).device)
        output = model(data)
    if hook is not None:
        hook.remove()
    return output
    return activations[0]


def create_sub_X(train_dataset, test_dataset, comb):
    # Extract tensors from Subset datasets
    test_X = torch.stack([test_dataset.dataset[i][0] for i in test_dataset.indices])
    train_X = torch.stack([train_dataset.dataset[i][0] for i in train_dataset.indices])
    sub_X = test_X.clone()
    unknown = list(set(range(train_X.shape[1])) - set(comb))
    rand_idxs = torch.randint(0, len(train_X), (test_X.shape[0],))
    rand_samples = train_X[rand_idxs]
    sub_X[:, unknown] = rand_samples[:, unknown]
    return sub_X
