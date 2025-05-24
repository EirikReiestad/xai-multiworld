import json
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def get_decision_tree_completeness_score(
    concept_scores_train,
    all_train_targets,
    concept_scores_test,
    all_test_targets,
    M,
    max_depth,
    iteration,
    result_path,
    suffix: str = "",
):
    print(f"  Testing with max_depth={max_depth}")
    model_dt = DecisionTreeClassifier(max_depth=max_depth)
    model_dt.fit(concept_scores_train, all_train_targets)

    feature_importances = model_dt.feature_importances_
    feature_split_counts = np.zeros(M)

    def count_feature_splits(node):
        if node == -1:
            return
        feature = model_dt.tree_.feature[node]
        if feature != -2:
            feature_split_counts[feature] += 1
            count_feature_splits(model_dt.tree_.children_left[node])
            count_feature_splits(model_dt.tree_.children_right[node])

    count_feature_splits(0)

    res = {}
    for m, importance, split in zip(
        range(M), feature_importances, feature_split_counts
    ):
        res[m] = (float(importance), int(split))

    with open(
        f"{result_path}/feature_importances{suffix}_{max_depth}_{iteration}.json",
        "w",
    ) as f:
        json.dump(res, f, indent=4)

    y_pred = model_dt.predict(concept_scores_test)
    accuracy = float(accuracy_score(all_test_targets, y_pred))
    print(
        f"  MNIST DecisionTree Test accuracy (M={M}, max_depth={max_depth}): {accuracy:.4f}"
    )
    return accuracy, res


def get_baseline_decision_tree_completeness_score(
    all_train_X,
    all_train_targets,
    all_test_X,
    all_test_targets,
    max_depth_values,
    results,
    result_path,
):
    X_train_flatten = all_train_X.reshape(all_train_X.shape[0], -1)
    y_train = all_train_targets.numpy()
    X_test_flatten = all_test_X.reshape(all_test_X.shape[0], -1)

    for max_depth in max_depth_values:
        print(f"\nBaseline test with max_depth={max_depth}")
        model_baseline = DecisionTreeClassifier(max_depth=max_depth)
        model_baseline.fit(X_train_flatten, y_train)

        y_pred = model_baseline.predict(X_test_flatten)
        baseline_accuracy = float(accuracy_score(all_test_targets, y_pred))
        print(
            f"Perfect info MNIST DecisionTree Test accuracy (max_depth={max_depth}): {baseline_accuracy:.4f}"
        )

        # Store baseline results
        result = {
            "M": "baseline",
            "max_depth": max_depth,
            "accuracy": baseline_accuracy,
            "similarity_matrix": None,
        }
        results.append(result)

        with open(
            f"{result_path}/baseline_decision_tree_{max_depth}.json",
            "w",
        ) as f:
            json.dump(result, f, indent=4)
