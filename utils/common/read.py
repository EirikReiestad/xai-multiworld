import json
import ast
from typing import Dict


def read_results(path: str) -> Dict:
    with open(path, "r") as f:
        results = json.load(f)
        results = {ast.literal_eval(key): value for key, value in results.items()}
    return results
