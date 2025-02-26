from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class ModelArtifact:
    model_weights: Mapping[str, Any]
    metadata: Dict[str, Any]
