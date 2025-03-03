import logging
from typing import Dict, List, Tuple

import numpy as np
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression

from utils.common.observation import Observation, load_and_split_observation
from utils.core.model_loader import ModelLoader
from xailib.core.linear_probing.linear_probe import LinearProbe


def get_probes(
    models: Dict[str, nn.Module],
    positive_observation: Observation,
    negative_observation: Observation,
    ignore: List["str"] = [],
) -> Tuple[Dict[str, Dict[str, LogisticRegression]], Dict, Dict]:
    regressors = {}
    positive_activations = {}
    negative_activations = {}

    positive_observation[..., Observation.LABEL] = 1
    negative_observation[..., Observation.LABEL] = 0

    for model_name, model in models.items():
        linear_probe = LinearProbe(
            model,
            positive_observation,
            negative_observation,
            ignore,
        )
        model_regressors, positive_activation, negative_activation = (
            linear_probe.train()
        )
        regressors[model_name] = model_regressors
        positive_activations[model_name] = positive_activation
        negative_activations[model_name] = negative_activation

    return regressors, positive_activations, negative_activations


def get_probe(
    concept: str,
    layer_idx: int,
    model: nn.Module,
    split: float = 0.8,
    ignore_layers: List[str] = [],
):
    model = ModelLoader.load_latest_model_from_path("artifacts", model)
    models = {"latest": model}
    positive_observation, test_observation = load_and_split_observation(concept, split)
    negative_observation, _ = load_and_split_observation("negative_" + concept, split)

    if len(positive_observation) == 0:
        logging.warning(f"Positive observation for {concept} is empty.")
        return None

    probes, positive_activations, negative_activations = get_probes(
        models, positive_observation, negative_observation, ignore_layers
    )

    probe = list(probes["latest"].values())[layer_idx]

    return probe
