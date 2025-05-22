import itertools
import json
import os

import numpy as np

from experiments.src.cavs import get_cavs
from experiments.src.compute_statistics import calc_similarity_matrix
from experiments.src.concept_score_handler import calc_concept_scores
from experiments.src.data_handler import flatten_data_loader, prepare_data
from experiments.src.decision_tree_handler import (
    get_baseline_decision_tree_completeness_score,
    get_decision_tree_completeness_score,
)
from experiments.src.elasticnet_handler import get_elasticnet_completeness_score
from experiments.src.logistic_regression import (
    get_logistic_regression_completeness_score,
)
from experiments.src.model_handler import train_or_load_model
from experiments.src.network import Net
from experiments.src.nn_handler import (
    get_neural_network_completeness_score,
    get_neural_network_feature_importance,
)
from experiments.src.random_forest_handler import get_random_forest_completeness_score
from experiments.src.svm_handler import get_svm_completeness_score
from experiments.src.xgboost_handler import get_xgboost_completeness_score


def main():
    # --- Settings ---
    batch_size = 32
    test_batch_size = 1000
    epochs = 10
    lr = 1e-1
    gamma = 0.7
    log_interval = 10
    dry_run = False
    load_model = True
    dataset_name = "mnist"
    model_name = f"experiments/{dataset_name}.pt"

    os.makedirs("experiments/results", exist_ok=True)

    # --- Data Preparation ---
    train_loader, test_loader = prepare_data(dataset_name, batch_size, test_batch_size)
    output_size = 10

    # --- Model Loading/Training ---
    model = Net()
    model = train_or_load_model(
        model,
        model_name,
        train_loader,
        test_loader,
        lr,
        gamma,
        epochs,
        log_interval,
        dry_run,
        load_model,
    )

    # --- Experiment Parameters ---
    M_values = [15] * 5
    max_depth_values = [15]
    lambda_1 = 0.1
    lambda_2 = 0.1
    lambda_3 = 0.1
    batch_size = 128
    cav_lr = 1e-3
    cav_epochs = 1
    average_instance_of_each_class = 1
    total_number_of_instances = 10
    average_class_ratio = average_instance_of_each_class / total_number_of_instances
    layer_name = "fc1"
    iteration = 0

    # --- Flatten dataset for processing ---
    all_train_X, all_train_targets = flatten_data_loader(train_loader)
    all_test_X, all_test_targets = flatten_data_loader(test_loader)

    # --- Storage for all results ---
    results = []

    lambdas_1 = np.linspace(0, 1, 4)
    lambdas_2 = np.linspace(0, 1, 4)
    lambdas_3 = np.linspace(0, 1, 4)
    lambda_combinations = list(itertools.product(lambdas_1, lambdas_2, lambdas_3))
    lambda_combinations = iter([[0.1, 0.1, 0.1]])

    # --- Main Experiment Loop ---
    for M in M_values:
        for lambdas in lambda_combinations:
            lambda_1, lambda_2, lambda_3 = lambdas
            print(f"\nTesting with M={M}")
            K = int(batch_size * average_class_ratio / 2)

            # CAVs and positive obs
            average_positive_observations, positive_observations = get_cavs(
                model,
                layer_name,
                train_loader,
                test_loader,
                M,
                K,
                lambda_1,
                lambda_2,
                lambda_3,
                batch_size,
                cav_lr,
                cav_epochs,
                iteration,
            )

            # Similarity matrix
            similarity_matrix = calc_similarity_matrix(
                average_positive_observations, positive_observations
            )

            # Concept score vectors
            concept_scores_train = calc_concept_scores(
                average_positive_observations, all_train_X
            )
            concept_scores_test = calc_concept_scores(
                average_positive_observations, all_test_X
            )

            accuracy, res = get_neural_network_feature_importance(
                M,
                concept_scores_train,
                all_train_targets,
                output_size,
                log_interval,
                dry_run,
                gamma,
                batch_size,
                test_batch_size,
                epochs,
            )

            # Decision tree experiments
            decision_tree_accuracy = 0
            for max_depth in max_depth_values:
                accuracy, res = get_decision_tree_completeness_score(
                    concept_scores_train,
                    all_train_targets,
                    concept_scores_test,
                    all_test_targets,
                    M,
                    max_depth,
                    iteration,
                )

                results.append(
                    {
                        "M": M,
                        "max_depth": max_depth,
                        "accuracy": accuracy,
                        "similarity_matrix": similarity_matrix,
                    }
                )
                decision_tree_accuracy = max(decision_tree_accuracy, accuracy)

            xgboost_accuracy, res = get_xgboost_completeness_score(
                concept_scores_train,
                all_train_targets,
                concept_scores_test,
                all_test_targets,
                M,
                None,
                iteration,
            )

            random_forest_accuracy, res = get_random_forest_completeness_score(
                concept_scores_train,
                all_train_targets,
                concept_scores_test,
                all_test_targets,
                M,
                None,
                iteration,
            )

            elasticnet_accuracy, res = get_elasticnet_completeness_score(
                concept_scores_train,
                all_train_targets,
                concept_scores_test,
                all_test_targets,
                M,
                iteration,
            )

            svm_linear_accuracy, res = get_svm_completeness_score(
                concept_scores_train,
                all_train_targets,
                concept_scores_test,
                all_test_targets,
                M,
                iteration,
            )

            svm_rbf_accuracy, res = get_svm_completeness_score(
                concept_scores_train,
                all_train_targets,
                concept_scores_test,
                all_test_targets,
                M,
                iteration,
            )

            logistic_regression_accuracy, res = (
                get_logistic_regression_completeness_score(
                    concept_scores_train,
                    all_train_targets,
                    concept_scores_test,
                    all_test_targets,
                    M,
                    iteration,
                )
            )

            # Baseline neural net experiment
            nn_accuracy, res = get_neural_network_completeness_score(
                M,
                concept_scores_train,
                all_train_targets,
                batch_size,
                test_batch_size,
                log_interval,
                dry_run,
                gamma,
                iteration,
            )

            with open("experiments/results/cav_experiment_results.json", "w") as f:
                json.dump(results, f, indent=4)

            os.makedirs("experiments/results/accuracies", exist_ok=True)

            with open(
                f"experiments/results/accuracies/{lambda_1}_{lambda_2}_{lambda_3}_{iteration}.json",
                "w",
            ) as f:
                accuracies = {
                    "decision_tree": decision_tree_accuracy,
                    "xgboost": xgboost_accuracy,
                    "random_forest": random_forest_accuracy,
                    "elasticnet": elasticnet_accuracy,
                    "svm_linear": svm_linear_accuracy,
                    "svm_rbf": svm_rbf_accuracy,
                    "logistic_regesssion": logistic_regression_accuracy,
                    "nn": nn_accuracy,
                }
                json.dump(accuracies, f, indent=4)

            iteration += 1

        # Baseline perfect info decision tree
        get_baseline_decision_tree_completeness_score(
            all_train_X,
            all_train_targets,
            all_test_X,
            all_test_targets,
            max_depth_values,
            results,
        )

        # Save all results
        with open("experiments/results/cav_experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults have been saved to json_results/cav_experiment_results.json")


if __name__ == "__main__":
    main()
