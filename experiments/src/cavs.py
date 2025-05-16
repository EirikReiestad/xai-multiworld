import datetime
import logging
import os
import time
from itertools import count

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from experiments.src.compute_statistics import calculate_statistics
from experiments.src.file_handler import store_images
from experiments.src.nn_handler import calculate_weights, get_activations
from torch.nn import functional as F
from tqdm import tqdm


def get_cavs(
    model,
    layer_name,
    train_loader,
    test_loader,
    M,
    K,
    lambda_1,
    lambda_2,
    lambda_3,
    batch_size,
    cav_lr,
    cav_epochs,
    iteration=0,
):
    (
        cavs,
        positive_observations,
        positive_activations,
        negative_observations,
        similarity_weights,
    ) = calculate_cavs(
        model=model,
        layer_name=layer_name,
        train_loader=train_loader,
        test_loader=test_loader,
        M=M,
        K=K,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        batch_size=batch_size,
        lr=cav_lr,
        epochs=cav_epochs,
    )

    probes = {}
    for i, name in enumerate(positive_activations.keys()):
        clf = LogisticRegression()
        clf.coef_ = cavs[i].detach().numpy()
        clf.intercept_ = np.array([0.0])
        probes[str(i)] = clf

    stats = calculate_statistics(
        list(positive_activations.keys()),
        positive_activations,
        probes,
        filename=f"cav_statistics_{iteration}.json",
    )

    weights = similarity_weights
    if "conv" in layer_name:
        weights = calculate_weights(
            model=model,
            cavs=cavs,
            layer_name=layer_name,
            data=positive_observations,
        )

    average_positive_observations = {}
    for key, value in positive_observations.items():
        if iteration == 0:
            store_images(
                list(value), f"experiments/tmp/average_positive_observations/{key}"
            )
        avg_obs = torch.stack([v * weights[key][i] for i, v in enumerate(value)]).mean(
            dim=0
        )
        average_positive_observations[key] = avg_obs
    if iteration == 0:
        store_images(
            list(average_positive_observations.values()), "experiments/tmp/avg_obs"
        )

    return average_positive_observations, positive_observations


def calculate_cavs(
    model: nn.Module,
    layer_name: str,
    train_loader,
    test_loader,
    M: int,
    K: int,
    lambda_1: float,
    lambda_2: float,
    lambda_3: float,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 10,
    log_frequency: int = 10,
    max_steps: int = 1000,
    convergence_threshold: float = 0.9,
    num_observations: int = 10000,
    num_sample_observations: int = 200,
    stats_filename: str = "cav_training_stats.json",
    result_path: str = "experiments/results",
):
    assert batch_size > K, "Batch size must be greater than K"

    stats = CAVTrainingStats()
    logging.info(
        f"Starting CAV training with {M} concepts, lambda_1={lambda_1}, lambda_2={lambda_2}"
    )

    data_sample = next(iter(train_loader))[0]
    activations = get_activations(model, data_sample, layer_name)

    feature_dim = activations[0].flatten().shape[-1]

    logging.info(f"Feature dimension: {feature_dim}")

    cavs = torch.randn(M, feature_dim)
    cavs = F.normalize(cavs, p=2, dim=1)
    cavs.requires_grad_(True)

    optimizer = torch.optim.Adam([cavs], lr=lr)

    steps_iterator = (
        tqdm(range(max_steps), desc="CAV Training")
        if max_steps < float("inf")
        else count()
    )

    for step in steps_iterator:
        batch = next(iter(train_loader))
        batch_data = batch[0]
        labels = batch[1]
        acts = get_activations(model, batch_data, layer_name)
        acts = F.normalize(acts, p=2, dim=1)
        acts = acts.view(acts.size(0), -1)
        similarity_matrix = torch.matmul(acts, cavs.T)

        top_k_indices = []
        for m in range(M):
            _, indices = torch.topk(similarity_matrix[:, m], K)
            top_k_indices.append(indices)

        top_k_indices = torch.stack(top_k_indices, dim=1)

        avg_objective = 0
        avg_coherence = 0
        avg_separation = 0
        avg_information_gain = 0

        grad_norm = float("nan")
        for epoch in range(epochs):
            objective, coherence, separation, information_gain, grad_norm = (
                optimize_cavs_with_stats(
                    cavs,
                    acts,
                    labels,
                    top_k_indices,
                    optimizer,
                    lambda_1,
                    lambda_2,
                    lambda_3,
                )
            )

            avg_objective += objective
            avg_coherence += coherence
            avg_separation += separation
            avg_information_gain += information_gain

            # After each epoch, re-normalize the CAVs to keep them unit length
            with torch.no_grad():
                cavs.data = F.normalize(cavs.data, p=2, dim=1)

        # Calculate averages over epochs
        avg_objective /= epochs
        avg_coherence /= epochs
        avg_separation /= epochs
        avg_information_gain /= epochs

        # Update and log statistics
        stats.update(
            step=step,
            epoch=epochs,
            objective=avg_objective,
            coherence=avg_coherence,
            separation=avg_separation,
            information_gain=avg_information_gain,
            grad_norm=grad_norm,
        )
        stats.log_progress(step, epochs, frequency=log_frequency)

        if step > 0 and avg_objective > convergence_threshold:
            stats.log_progress(step, epochs, frequency=1)
            logging.info(
                f"Converged at step {step} with objective value {avg_objective:.4f}"
            )
            break

        if step >= max_steps - 1:
            logging.info(f"Reached maximum steps ({max_steps})")
            break

    stats.save_stats(filename=stats_filename, directory=result_path)

    test_batch = next(iter(test_loader))
    test_data = test_batch[0]
    acts = get_activations(model, test_data, layer_name)
    acts = F.normalize(acts, p=2, dim=1)
    acts = acts.view(acts.size(0), -1)
    similarity_matrix = torch.matmul(acts, cavs.T)  # [batch_size, M]

    positive_observations = {}
    positive_activations = {}
    negative_observations = {}
    negative_activation_data = {}
    similarity_weights = {}
    for m in range(M):
        k = min(num_observations, num_sample_observations)
        _, indices = torch.topk(similarity_matrix[:, m], k)
        m_observation = test_data[indices]
        positive_observations[str(m)] = m_observation
        positive_activations[str(m)] = acts[indices]
        similarity_weights[str(m)] = (
            similarity_matrix[indices, m].detach().cpu().numpy()
        )
        """
        filename = f"{m}_cav_positive_observations.json"
        path = os.path.join(result_path, filename)
        with open(path, "w") as f:
            json.dump(m_observation, f, indent=4)

        negative_indices = [i for i in range(k) if i not in indices]
        negative_observation = test_batch[negative_indices][:num_sample_observations]
        negative_observations[str(m)] = negative_observation

        filename = f"{m}_cav_negative_observations.json"
        path = os.path.join(result_path, filename)
        with open(path, "w") as f:
            json.dump(negative_observation, f, indent=4)
        """

    return (
        cavs,
        positive_observations,
        positive_activations,
        negative_observations,
        similarity_weights,
    )


class CAVTrainingStats:
    """Tracks and logs statistics for CAV training"""

    def __init__(self):
        self.stats = {
            "step": [],
            "epoch": [],
            "objective_value": [],
            "coherence_term": [],
            "separation_term": [],
            "information_gain_term": [],
            "grad_norm": [],
            "timestamp": [],
            "elapsed_time": [],
        }
        self.start_time = time.time()

    def update(
        self,
        step: int,
        epoch: int,
        objective: float,
        coherence: float,
        separation: float,
        information_gain: float,
        grad_norm: float | None = None,
    ):
        """Record stats for a training step"""
        now = time.time()
        self.stats["step"].append(step)
        self.stats["epoch"].append(epoch)
        self.stats["objective_value"].append(objective)
        self.stats["coherence_term"].append(coherence)
        self.stats["separation_term"].append(separation)
        self.stats["information_gain_term"].append(information_gain)
        self.stats["grad_norm"].append(
            grad_norm if grad_norm is not None else float("nan")
        )
        self.stats["timestamp"].append(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.stats["elapsed_time"].append(now - self.start_time)

    def log_progress(self, step: int, epoch: int, frequency: int = 10):
        """Log progress at specified frequency"""
        if step % frequency == 0:
            idx = -1  # Get the latest stats

            # Format log message
            log_msg = (
                f"Step {self.stats['step'][idx]:04d} | "
                f"Epoch {self.stats['epoch'][idx]:02d} | "
                f"Objective: {self.stats['objective_value'][idx]:.4f} | "
                f"Coherence: {self.stats['coherence_term'][idx]:.4f} | "
                f"Separation: {self.stats['separation_term'][idx]:.4f} | "
                f"Information gain: {self.stats['information_gain_term'][idx]:.4f}"
            )

            if not np.isnan(self.stats["grad_norm"][idx]):
                log_msg += f" | Grad Norm: {self.stats['grad_norm'][idx]:.4f}"

            log_msg += f" | Time: {self.stats['elapsed_time'][idx]:.1f}s"

            # Log the message
            print(log_msg)

    def to_dataframe(self):
        """Convert stats to pandas DataFrame (if pandas is available)"""
        try:
            import pandas as pd

            return pd.DataFrame(self.stats)
        except ImportError:
            logging.warning("pandas not available. Install pandas to use this feature.")
            return self.stats

    def save_stats(
        self,
        directory: str = "experiments/results",
        filename: str = "cav_training_stats.json",
    ):
        """Save stats to CSV file"""
        try:
            df = self.to_dataframe()
            if hasattr(df, "to_csv"):
                filepath = os.path.join(directory, filename)
                df.to_json(filepath, index=False)
                logging.info(f"Training stats saved to {filename}")
            else:
                logging.warning("Could not save stats. pandas is required.")
        except Exception as e:
            logging.error(f"Failed to save stats: {e}")


def optimize_cavs_with_stats(
    cavs: torch.Tensor,
    data: torch.Tensor,
    labels: torch.Tensor,
    top_k_indices: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lambda_1: float,
    lambda_2: float,
    lambda_3: float,
):
    """Optimize CAVs and return detailed statistics about the optimization step"""
    optimizer.zero_grad()

    coherence_term = 0
    for m in range(cavs.shape[0]):
        top_k_data = data[top_k_indices[:, m]]
        similarities = torch.sum(torch.matmul(top_k_data, cavs[m]))
        coherence_term += similarities

    cav_similarities = torch.matmul(cavs, cavs.T)
    mask = 1.0 - torch.eye(cavs.shape[0], device=cavs.device)
    separation_term = torch.sum(torch.abs(cav_similarities * mask))

    information_gain_term = 0
    for m in range(cavs.shape[0]):
        top_k_data = data[top_k_indices[:, m]]
        top_k_labels = labels[top_k_indices[:, m]]
        unique_labels, counts = torch.unique(top_k_labels, return_counts=True)
        max_count = max(counts)
        information_gain_term += max_count / top_k_labels.numel()

        # If I want to weight by distance
        """
        pointwise_sims = torch.matmul(top_k_data, cavs[m])
        majority_label = unique_labels[counts.argmax()]
        reward = (
            pointwise_sims[top_k_labels == majority_label].sum() / top_k_labels.numel()
        )
        information_gain_term += reward
        """

    objective = (
        lambda_1 * coherence_term
        - lambda_2 * separation_term
        + lambda_3 * information_gain_term
    )

    loss = -objective

    loss.backward()

    grad_norm = cavs.grad.norm().item() if cavs.grad is not None else float("nan")

    if torch.isnan(cavs.grad).any() or torch.isinf(cavs.grad).any():
        logging.warning("NaN or Inf detected in gradients!")

    optimizer.step()

    return (
        objective.item(),
        coherence_term.item(),
        separation_term.item(),
        information_gain_term.item(),
        grad_norm,
    )
