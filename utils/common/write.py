import json
from typing import Dict


def write_results(results: Dict, path: str, custom_cls=None):
    with open(path, "w") as f:
        results = {str(key): value for key, value in results.items()}
        json.dump(results, f, cls=custom_cls)
