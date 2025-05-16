import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_logistic_regression_completeness_score(
    concept_scores_train,
    all_train_targets,
    concept_scores_test,
    y_true,
    M,
    iteration,
    suffix="",
):
    print("Testing Logistic Regression")
    model_lr = LogisticRegression(max_iter=10000)
    model_lr.fit(concept_scores_train, all_train_targets)

    # Logistic Regression feature importances (absolute value of coefficients)
    feature_importances = np.abs(model_lr.coef_).sum(axis=0)
    res = {}
    for m, importance in zip(range(M), feature_importances):
        res[m] = (float(importance), 0)  # No split count for Logistic Regression

    os.makedirs("experiments/results", exist_ok=True)
    with open(
        f"experiments/results/logistic_regression_feature_importances{suffix}_{iteration}.json",
        "w",
    ) as f:
        json.dump(res, f, indent=4)

    y_pred = model_lr.predict(concept_scores_test)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"  Logistic Regression Test Accuracy: {accuracy:.4f}")
    return accuracy, res
