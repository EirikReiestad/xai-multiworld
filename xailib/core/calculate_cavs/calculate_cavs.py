import json
import logging
import os
import time
from datetime import datetime
from itertools import count
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # For notebook progress bars (optional)

from multiworld.base import MultiWorldEnv
from utils.common.collect_rollouts import collect_rollouts
from utils.common.model_artifact import ModelArtifact
from utils.common.numpy_collections import NumpyEncoder
from utils.common.observation import Observation
from xailib.utils.activations import get_activations

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CAV-Training")

# Add file handler if you want to save logs to file
# file_handler = logging.FileHandler("cav_training.log")
# file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
# logger.addHandler(file_handler)

torch.autograd.set_detect_anomaly(True)


class CAVTrainingStats:
    """Tracks and logs statistics for CAV training"""

    def __init__(self):
        self.stats = {
            "step": [],
            "epoch": [],
            "objective_value": [],
            "coherence_term": [],
            "separation_term": [],
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
        grad_norm: Optional[float] = None,
    ):
        """Record stats for a training step"""
        now = time.time()
        self.stats["step"].append(step)
        self.stats["epoch"].append(epoch)
        self.stats["objective_value"].append(objective)
        self.stats["coherence_term"].append(coherence)
        self.stats["separation_term"].append(separation)
        self.stats["grad_norm"].append(
            grad_norm if grad_norm is not None else float("nan")
        )
        self.stats["timestamp"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
                f"Separation: {self.stats['separation_term'][idx]:.4f}"
            )

            if not np.isnan(self.stats["grad_norm"][idx]):
                log_msg += f" | Grad Norm: {self.stats['grad_norm'][idx]:.4f}"

            log_msg += f" | Time: {self.stats['elapsed_time'][idx]:.1f}s"

            # Log the message
            logger.info(log_msg)

    def to_dataframe(self):
        """Convert stats to pandas DataFrame (if pandas is available)"""
        try:
            import pandas as pd

            return pd.DataFrame(self.stats)
        except ImportError:
            logger.warning("pandas not available. Install pandas to use this feature.")
            return self.stats

    def save_stats(
        self,
        filepath: str = os.path.join("assets", "results"),
        filename: str = "cav_training_stats.csv",
    ):
        """Save stats to CSV file"""
        try:
            df = self.to_dataframe()
            if hasattr(df, "to_csv"):
                filename = os.path.join(filepath, filename)
                df.to_csv(filepath, index=False)
                logger.info(f"Training stats saved to {filename}")
            else:
                logger.warning("Could not save stats. pandas is required.")
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")


def calculate_cavs(
    model: nn.Module,
    env: MultiWorldEnv,
    artifact: ModelArtifact,
    method: Literal["policy", "random"],
    M: int,
    K: int,
    lambda_1: float,
    lambda_2: float,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 10,
    ignore_layers: list = [],
    log_frequency: int = 10,
    max_steps: int = 1000,
    convergence_threshold: float = 0.9,
    save_stats: bool = True,
    stats_filename: str = "cav_training_stats.csv",
    observation_path: str = "assets/observations",
    save_observations: bool = True,
):
    """
    Train Concept Activation Vectors (CAVs)

    Args:
        model: Neural network model
        env: Environment for collecting rollouts
        artifact: Model artifact for rollouts
        method: Method for collecting rollouts ("policy" or "random")
        M: Number of concepts/CAVs to train
        K: Number of top observations to use for each concept
        lambda_1: Weight for coherence term (positive)
        lambda_2: Weight for separation term (negative)
        batch_size: Batch size for training
        lr: Learning rate
        epochs: Number of epochs per step
        ignore_layers: Layers to ignore when getting activations
        log_frequency: How often to log progress
        max_steps: Maximum number of steps before stopping
        convergence_threshold: Objective value threshold for convergence
        save_stats: Whether to save statistics to file
        stats_filename: Filename for saved statistics

    Returns:
        Trained CAVs and training statistics
    """
    assert batch_size > K, "Batch size must be greater than K"

    # Initialize stats tracker
    stats = CAVTrainingStats()

    # Log training configuration
    logger.info(
        f"Starting CAV training with {M} concepts, lambda_1={lambda_1}, lambda_2={lambda_2}"
    )

    # Get initial activations to determine feature dimension
    observation = collect_rollouts(
        env,
        artifact,
        1,
        method=method,
        observation_path=os.path.join("assets", "tmp"),
        force_update=True,
    )

    activations, input, output = get_activations(
        {"latest": model}, observation, ignore_layers=ignore_layers
    )
    latest_activations = activations["latest"]
    last_layer_activations = list(latest_activations.values())[-1]["output"]

    # Determine feature dimension
    if isinstance(last_layer_activations, np.ndarray):
        feature_dim = last_layer_activations.shape[-1]
    else:
        feature_dim = last_layer_activations.size(-1)

    logger.info(f"Feature dimension: {feature_dim}")

    # Initialize CAVs with normalized random vectors
    cavs = torch.randn(M, feature_dim)
    cavs = F.normalize(cavs, p=2, dim=1)  # Unit normalize each CAV
    cavs.requires_grad_(True)  # Make it explicitly require gradients

    # Use Adam optimizer with a reasonable learning rate
    optimizer = torch.optim.Adam([cavs], lr=lr)

    # Main training loop
    steps_iterator = (
        tqdm(range(max_steps), desc="CAV Training")
        if max_steps < float("inf")
        else count()
    )

    for step in steps_iterator:
        # Collect new batch of observations
        observation = collect_rollouts(
            env,
            artifact,
            batch_size,
            method=method,
            observation_path=os.path.join("assets", "tmp"),
            force_update=True,
        )

        # Get activations for the new batch
        activations, input, output = get_activations(
            {"latest": model}, observation, ignore_layers=ignore_layers
        )
        latest_activations = activations["latest"]
        last_layer_values = list(latest_activations.values())[-1]["output"]

        # Convert to tensor if needed and normalize
        if isinstance(last_layer_values, np.ndarray):
            data = torch.tensor(last_layer_values, dtype=torch.float32)
        else:
            data = (
                last_layer_values.detach().clone()
            )  # Detach to avoid backprop through model

        # Normalize data
        data = F.normalize(data, p=2, dim=1)  # Normalize each data point

        # Pre-compute similarity matrix once for this batch
        similarity_matrix = torch.matmul(data, cavs.T)  # [batch_size, M]

        # Get top K indices for each concept
        top_k_indices = []
        for m in range(M):
            # Get the K data points most similar to this CAV
            _, indices = torch.topk(similarity_matrix[:, m], K)
            top_k_indices.append(indices)

        top_k_indices = torch.stack(top_k_indices, dim=1)  # [K, M]

        # Training for multiple epochs
        avg_objective = 0
        avg_coherence = 0
        avg_separation = 0

        for epoch in range(epochs):
            # Optimize CAVs for one step
            objective, coherence, separation, grad_norm = optimize_cavs_with_stats(
                cavs, data, top_k_indices, optimizer, lambda_1, lambda_2
            )

            avg_objective += objective
            avg_coherence += coherence
            avg_separation += separation

            # After each epoch, re-normalize the CAVs to keep them unit length
            with torch.no_grad():
                cavs.data = F.normalize(cavs.data, p=2, dim=1)

        # Calculate averages over epochs
        avg_objective /= epochs
        avg_coherence /= epochs
        avg_separation /= epochs

        # Update and log statistics
        stats.update(
            step=step,
            epoch=epochs,
            objective=avg_objective,
            coherence=avg_coherence,
            separation=avg_separation,
            grad_norm=grad_norm,
        )
        stats.log_progress(step, epochs, frequency=log_frequency)

        # Check for convergence
        if step > 0 and avg_objective > convergence_threshold:
            stats.log_progress(step, epochs, frequency=1)
            logger.info(
                f"Converged at step {step} with objective value {avg_objective:.4f}"
            )
            break

        # Check if we've reached maximum steps
        if step >= max_steps - 1:
            logger.info(f"Reached maximum steps ({max_steps})")
            break

    # Save stats if requested
    if save_stats:
        stats.save_stats(stats_filename)

    if save_observations:
        observation = collect_rollouts(
            env,
            artifact,
            10000,
            method=method,
            observation_path=os.path.join("assets", "tmp"),
            force_update=True,
        )
        observation_data = observation[..., Observation.OBSERVATION]

        activations, input, output = get_activations(
            {"latest": model}, observation, ignore_layers=ignore_layers
        )
        latest_activations = activations["latest"]
        last_layer_values = list(latest_activations.values())[-1]["output"]

        if isinstance(last_layer_values, np.ndarray):
            data = torch.tensor(last_layer_values, dtype=torch.float32)
        else:
            data = (
                last_layer_values.detach().clone()
            )  # Detach to avoid backprop through model

        similarity_matrix = torch.matmul(data, cavs.T)  # [batch_size, M]

        for m in range(M):
            _, indices = torch.topk(similarity_matrix[:, m], 100)
            m_observation = [obs[0] for obs in observation_data[indices]]
            filename = f"{m}_cav_observations.json"
            path = os.path.join(observation_path, filename)
            with open(path, "w") as f:
                json.dump(m_observation, f, indent=4, cls=NumpyEncoder)

    return cavs, stats


def optimize_cavs_with_stats(
    cavs: torch.Tensor,
    data: torch.Tensor,
    top_k_indices: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lambda_1: float,
    lambda_2: float,
):
    """Optimize CAVs and return detailed statistics about the optimization step"""
    optimizer.zero_grad()

    # Calculate coherence term (we want to maximize this)
    coherence_term = 0
    for m in range(cavs.shape[0]):
        # Get the K data points most similar to this CAV
        top_k_data = data[top_k_indices[:, m]]
        # Calculate similarities and sum them
        similarities = torch.sum(torch.matmul(top_k_data, cavs[m]))
        coherence_term += similarities

    # Calculate separation term (we want to minimize this)
    # Compute pairwise similarities between CAVs
    cav_similarities = torch.matmul(cavs, cavs.T)
    # Create a mask to exclude self-similarities
    mask = 1.0 - torch.eye(cavs.shape[0], device=cavs.device)
    # Sum all pairwise similarities (excluding self-similarities)
    separation_term = torch.sum(torch.abs(cav_similarities * mask))

    # Our objective (what we want to maximize)
    objective = lambda_1 * coherence_term - lambda_2 * separation_term

    # For maximization with PyTorch optimizers, negate the objective
    loss = -objective  # This is what we'll actually minimize

    # Backpropagate
    loss.backward()

    # Calculate gradient norm for monitoring
    grad_norm = cavs.grad.norm().item() if cavs.grad is not None else float("nan")

    # Check for gradient issues
    if torch.isnan(cavs.grad).any() or torch.isinf(cavs.grad).any():
        logger.warning("NaN or Inf detected in gradients!")

    # Update parameters
    optimizer.step()

    # Return detailed statistics
    return objective.item(), coherence_term.item(), separation_term.item(), grad_norm
