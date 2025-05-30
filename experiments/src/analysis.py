import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from experiments.src.compute_statistics import (
    collect_accuracies,
    collect_and_compute_agreement,
    collect_and_compute_variance,
    collect_max_accuracy,
)
from experiments.src.file_handler import read_multi_files


def calculate_statistics(
    df, save_dir: str = "experiments/results", save_filename: str = "statistics"
):
    df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    dropped_columns = [
        "mean",
        "median",
        "variance",
        "std_dev",
        # "iq_range",
    ]
    df = df.drop(dropped_columns, axis=1)

    correlation_matrix = df.corr()
    print(df.head())
    print(correlation_matrix)

    corrs = df.drop(columns=["importance"]).corrwith(df["importance"])
    print("Individual metric correlations:\n", corrs)

    X = df.drop(columns=["importance"])
    y = df["importance"]
    X = sm.add_constant(X)  # optional, adds intercept

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

    model = sm.OLS(y, X).fit()
    print(model.summary())
    result = sm.OLS(y, X).fit()
    latex = result.summary2().as_latex()

    path = os.path.join(save_dir, f"{save_filename}.tex")
    with open(path, "w") as f:
        f.write(latex)


def main(base_filenames: list[str], result_dir: str):
    n = 20
    df_statistics = read_multi_files(
        filename="cav_statistics", n=n, result_dir=result_dir
    )

    for base in base_filenames:
        print(f"Statistics for {base}")
        df_importance = read_multi_files(filename=base, n=n, result_dir=result_dir)
        df_importance.columns = ["importance", "splits"]
        df = pd.concat([df_statistics, df_importance], axis=1)
        calculate_statistics(
            df, save_dir=result_dir, save_filename=f"results_statistics_{base}"
        )

    collect_and_compute_variance(base_filenames, result_dir)
    collect_and_compute_agreement(base_filenames, result_dir, method="rbo")
    collect_and_compute_agreement(base_filenames, result_dir, method="topk")
    # collect_max_accuracy(base_filenames, results_dir=result_dir, num_iterations=n)
    # collect_accuracies(
    #     results_dir=os.path.join(result_dir, "accuracies"), output_dir=result_dir
    # )


if __name__ == "__main__":
    base_filenames = [
        "elasticnet_feature_importances",
        "feature_importances_15",
        "randomforest_feature_importances",
        "xgboost_feature_importances",
        "logistic_regression_feature_importances",
        "svm_linear_feature_importances",
    ]
    main(base_filenames, "experiments/results_completeness_15")
    # main(base_filenames, "experiments/results_completeness_lambda_0_1_5_15")
    # main(base_filenames, "experiments/results_completeness_lambda_0_-1_5_15")
    # main(base_filenames, "experiments/results_completeness_lambda_-1_1_5_15")
    # main(base_filenames, "experiments/results_completeness_lambda_-1_1_10_15")
    # base_filenames.append("nn_feature_importances")
    # main(base_filenames, "experiments/results_importance_15")
