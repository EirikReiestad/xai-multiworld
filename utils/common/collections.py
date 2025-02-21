import itertools
import json
from itertools import chain
from typing import Any, Dict, List, TypeVar

T = TypeVar("T")


def flatten_dicts(values: List[Dict[Any, T]]) -> List[T]:
    return list(chain.from_iterable(value.values() for value in values))


def zip_dict_list(dict_list: list[dict[str, Any]]) -> List[List[Any]]:
    data = []
    for key in dict_list[0].keys():
        data.append([dict_list[i][key] for i in range(len(dict_list))])
    return data


def get_combinations_dict(data: Dict[Any, Any]) -> List[Dict[Any, Any]]:
    keys = list(data.keys())
    key_combinations = []
    for r in range(1, len(keys) + 1):
        key_combinations.extend(itertools.combinations(keys, r))

    result_combinations = []
    for key_pair in key_combinations:
        result_combinations.append({key: data[key] for key in key_pair})
    return result_combinations


def get_combinations(data: List):
    combinations = []
    for r in range(1, len(data) + 1):
        combinations.extend(itertools.combinations(data, r))

    result_combinations = [list(comb) for comb in combinations]
    return result_combinations


class DefaultEncoder(json.JSONEncoder):
    def default(self, o):
        return super().default(o)
