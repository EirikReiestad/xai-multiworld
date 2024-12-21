from typing import Mapping, Any
from utils.core.model_loader import ModelLoader
import os

path = os.path.join("artifacts")

model_loader = ModelLoader.load_from_path(path)
print(model_loader)
