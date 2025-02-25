import logging
from typing import Dict, List

from tabulate import tabulate


def log_similarity(similarities: Dict, concepts: List[str]):
    y_axis_values = sorted(next(iter(similarities.values())).keys(), reverse=True)

    table_data = []
    for y in y_axis_values:
        row = [y] + [similarities[key][y][0][0] for key in similarities.keys()]
        table_data.append(row)

    headers = ["dataset size"] + concepts
    logging.info("\n" + tabulate(table_data, headers=headers))


def log_stats(stats: Dict[str, Dict[str, float]]):
    metrics = list(next(iter(stats.values())).keys())
    concepts = list(stats.keys())

    table_data = []
    for metric in metrics:
        row = [metric] + [stats[concept][metric] for concept in concepts]
        table_data.append(row)

    table = tabulate(
        table_data,
        headers=["Metric"] + concepts,
    )
    logging.info(f"\n{table}")
