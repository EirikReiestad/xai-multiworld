import numpy as np


def all_same_values_in_dict(a: dict):
    first = list(a.values())[0]
    return all([value == first for value in a.values()])
