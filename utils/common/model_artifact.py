from dataclasses import dataclass
from typing import Any, Mapping, Dict
import torch.nn as nn


@dataclass
class ModelArtifact:
    model_weights: Mapping[str, Any]
    metadata: Dict[str, Any]
    model: nn.Module
