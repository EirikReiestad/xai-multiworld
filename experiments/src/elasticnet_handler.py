import json
import os

import numpy as np
from skimage.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score


def get_elasticnet_completeness_score(
    concept_scores_train,
    all_train_targets,
    concept_scores_test,
    y_true,
    M,
    iteration,
    alpha=0.01,
    l1_ratio=0.5,
    suffix="",
):
    print(f"  Testing ElasticNet with alpha={alpha}, l1_ratio={l1_ratio}")
    model_en = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    model_en.fit(concept_scores_train, all_train_targets)

    # ElasticNet feature importances (absolute value of coefficients)
    feature_importances = np.abs(model_en.coef_)
    res = {}
    for m, importance in zip(range(M), feature_importances):
        res[m] = (float(importance), 0)  # No split count for ElasticNet

    os.makedirs("experiments/results", exist_ok=True)
    with open(
        f"experiments/results/elasticnet_feature_importances{suffix}_{iteration}.json",
        "w",
    ) as f:
        json.dump(res, f, indent=4)

    y_pred = model_en.predict(concept_scores_test)
    mse = mean_squared_error(y_true.detach().numpy(), y_pred)
    y_pred = np.array([round(x) for x in y_pred])
    accuracy = accuracy_score(y_true, y_pred)
    print(
        f"  ElasticNet Test MSE (M={M}, alpha={alpha}, l1_ratio={l1_ratio}): {mse:.4f}, accuracy={accuracy:.4f}"
    )
    return accuracy, res
