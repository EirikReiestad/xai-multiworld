import logging
from typing import Dict, List

from tabulate import tabulate


def log_decision_tree_feature_importance(feature_importance):
    table_data = sorted(
        feature_importance.items(), key=lambda item: item[1][0], reverse=True
    )

    formatted_data = [
        (concept, f"{value[0]:.6f}", value[1]) for concept, value in table_data
    ]

    logging.info(
        "\n"
        + tabulate(formatted_data, headers=("Concept", "Feature importance", "Splits"))
    )


def log_shapley_values(shapley_values: Dict):
    table_data = dict(
        sorted(shapley_values.items(), key=lambda item: item[1], reverse=True)
    )
    logging.info(
        "\n" + tabulate(table_data.items(), headers=("Concept", "Shapley value"))
    )


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
