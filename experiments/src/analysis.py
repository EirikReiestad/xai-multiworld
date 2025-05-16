import statsmodels.api as sm
from src.compute_statistics import (
    collect_accuracies,
    collect_and_compute_variance,
)


def calculate_statistics(df):
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

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # plot_scatter(list(df["importance"]), list(df["mean"]), show=False)


def main():
    """
    n = 1
    df_statistics = read_multi_files(filename="cav_statistics", n=n)
    df_importance = read_multi_files(filename="feature_importances_15", n=n)
    df_importance.columns = ["importance", "splits"]
    df = pd.concat([df_statistics, df_importance], axis=1)

    calculate_statistics(df)
    """

    """
    base_filenames = [
        "feature_importances_15",
        "randomforest_feature_importances",
        "xgboost_feature_importances",
        "elasticnet_feature_importances",
    ]
    collect_and_compute_variance(base_filenames)
    collect_max_accuracy(base_filenames)
    """

    base_filenames = [
        "cav_experiment_accuracy_0.1",
        "cav_experiment_accuracy_0.5",
        "cav_experiment_accuracy_1",
    ]
    collect_accuracies(base_filenames)


if __name__ == "__main__":
    main()
