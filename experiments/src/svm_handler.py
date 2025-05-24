import json
import os
from typing import Literal

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def get_svm_completeness_score(
    concept_scores_train,
    all_train_targets,
    concept_scores_test,
    y_true,
    M,
    iteration,
    result_path,
    kernel: Literal["linear", "rbf"] = "linear",
    suffix="",
):
    print("  Testing SVM")
    model_svm = SVC(kernel=kernel, probability=True, max_iter=10000)
    model_svm.fit(concept_scores_train, all_train_targets)

    # SVM feature importances (absolute value of coefficients)
    feature_importances = np.abs(model_svm.coef_).sum(axis=0)
    res = {}
    for m, importance in zip(range(M), feature_importances):
        res[m] = (float(importance), 0)  # No split count for SVM

    with open(
        f"{result_path}/svm_{kernel}_feature_importances{suffix}_{iteration}.json",
        "w",
    ) as f:
        json.dump(res, f, indent=4)

    y_pred = model_svm.predict(concept_scores_test)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"  SVM Test Accuracy: {accuracy:.4f}")
    return accuracy, res
