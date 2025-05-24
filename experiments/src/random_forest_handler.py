import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def get_random_forest_completeness_score(
    concept_scores_train,
    all_train_targets,
    concept_scores_test,
    all_test_targets,
    M,
    max_depth,
    iteration,
    result_path,
    suffix="",
):
    print(f"  Testing Random Forest with max_depth={max_depth}")
    model_rf = RandomForestClassifier(
        n_estimators=100, max_depth=max_depth, random_state=42, n_jobs=-1
    )
    model_rf.fit(concept_scores_train, all_train_targets)

    feature_importances = model_rf.feature_importances_
    # For RandomForest, estimating split counts by summing from all trees
    feature_split_counts = np.zeros(M)
    for estimator in model_rf.estimators_:
        tree = estimator.tree_
        for feature in tree.feature:
            if feature >= 0:
                feature_split_counts[feature] += 1

    res = {}
    for m, importance, split in zip(
        range(M), feature_importances, feature_split_counts
    ):
        res[m] = (float(importance), int(split))

    with open(
        f"{result_path}/randomforest_feature_importances{suffix}_{iteration}.json",
        "w",
    ) as f:
        json.dump(res, f, indent=4)

    y_pred = model_rf.predict(concept_scores_test)
    accuracy = float(accuracy_score(all_test_targets, y_pred))
    print(
        f"  MNIST RandomForest Test accuracy (M={M}, max_depth={max_depth}): {accuracy:.4f}"
    )
    return accuracy, res
