from itertools import count
import json
import ast
from typing import Dict
import pandas as pd


def read_results(path: str) -> Dict:
    with open(path, "r") as f:
        results = json.load(f)
        results = {ast.literal_eval(key): value for key, value in results.items()}
    return results


def read_multi_files(path: str):
    dfs = []
    for i in count():
        try:
            df = pd.read_json(f"{path}_{i}.json").T
        except FileNotFoundError:
            break
        dfs.append(df)
    df = pd.concat(dfs)
    # df.index = range(len(df))
    return df
