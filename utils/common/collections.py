from typing import Dict, Any, List, TypeVar
from itertools import chain

T = TypeVar("T")


def flatten_dicts(values: List[Dict[Any, T]]) -> List[T]:
    return list(chain.from_iterable(value.values() for value in values))
