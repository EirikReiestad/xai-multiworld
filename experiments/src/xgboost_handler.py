import json
import os

import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb


def get_xgboost_completeness_score(
    concept_scores_train,
    all_train_targets,
    concept_scores_test,
    all_test_targets,
    M,
    max_depth,
    iteration,
    suffix: str = "",
):
    print(f"  Testing XGBoost with max_depth={max_depth}")
    model_xgb = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model_xgb.fit(concept_scores_train, all_train_targets)

    feature_importances = model_xgb.feature_importances_
    booster = model_xgb.get_booster()
    fmap = {f"f{idx}": idx for idx in range(M)}
    feature_split_counts = np.zeros(M)

    # Count feature splits from trees
    for tree in booster.get_dump(dump_format="json"):
        tree_json = json.loads(tree)
        nodes = [tree_json]
        while nodes:
            node = nodes.pop()
            if "split" in node:
                fidx = int(node["split"][1:])  # 'f0' -> 0
                feature_split_counts[fidx] += 1
                if "children" in node:
                    nodes.extend(node["children"])

    res = {}
    for m, importance, split in zip(
        range(M), feature_importances, feature_split_counts
    ):
        res[m] = (float(importance), int(split))

    os.makedirs("experiments/results", exist_ok=True)
    with open(
        f"experiments/results/xgboost_feature_importances{suffix}_{iteration}.json", "w"
    ) as f:experiments
        json.dump(res, f, indent=4)

    y_pred = model_xgb.predict(concept_scores_test)
    accuracy = float(accuracy_score(all_test_targets, y_pred))
    print(
        f"  MNIST XGBoost Test accuracy (M={M}, max_depth={max_depth}): {accuracy:.4f}"
    )
    return accuracy, res
