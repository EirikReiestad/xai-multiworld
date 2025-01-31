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


class DefaultEncoder(json.JSONEncoder):
    def default(self, o):
        return super().default(o)
