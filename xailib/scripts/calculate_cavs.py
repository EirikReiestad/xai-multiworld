import logging
import os

from joblib.pool import np
from sklearn.linear_model import LogisticRegression

from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.common.observation import filter_observations
from utils.core.model_loader import ModelLoader
from xailib.common.completeness_score import get_completeness_score
from xailib.core.calculate_cavs.calculate_cavs import calculate_cavs
from xailib.utils.metrics import (
    calculate_cav_similarity,
    calculate_probe_similarities,
    calculate_statistics,
)


def main():
    artifact_path = "artifacts"
    model_type = "dqn"
    eval = True
    ignore_layers = ["_fc0"]
    result_path = os.path.join("assets", "results")
    method = "random"

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=eval,
        artifact_path=artifact_path,
    )
    model = list(models.values())[-1]

    M = 10
    lambda_1 = 0.1
    lambda_2 = 0.1
    lambda_3 = 0.1
    batch_size = 128
    lr = 1e-3
    epochs = 1

    average_instance_of_each_class = 100
    total_number_of_instances = 1000
    average_class_ratio = average_instance_of_each_class / total_number_of_instances
    K = int(batch_size * average_class_ratio / 2)

    (
        cavs,
        stats,
        positive_observations,
        negative_observations,
        positive_activations,
        negative_observations,
        similarity_weights,
    ) = calculate_cavs(
        model=model,
        env=environment,
        artifact=artifact,
        method=method,
        M=M,
        K=K,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        num_observations=1000,
        num_sample_observations=200,
        ignore_layers=ignore_layers,
    )

    cavs = cavs.detach().cpu().numpy()
    random_cav = np.random.randn(1, cavs[0].shape[0])
    cavs = {f"{i}": cav for i, cav in enumerate(cavs)}
    mock_probes = {}

    mock_probe = LogisticRegression()
    mock_probe.coef_ = random_cav
    mock_probe.intercept_ = 0
    mock_probe.classes_ = [0, 1]
    mock_probes["random"] = {"latest": {"layer": mock_probe}}

    # logging.info("Visualizing CAVs using Lucent...")
    # lucid_visualization(model, cavs)

    logging.info("Calculating cav similarity...")
    calculate_cav_similarity(cavs, result_path, "cav_similarity.json")
    for i, cav in cavs.items():
        mock_probe = LogisticRegression()
        mock_probe.coef_ = cav
        mock_probe.intercept_ = 0
        mock_probe.classes_ = [0, 1]
        mock_probes[str(i)] = {"latest": {"layer": mock_probe}}

    cav_names = list(mock_probes.keys())

    observations = collect_rollouts(
        environment,
        artifact,
        1000,
        method=method,
        observation_path=os.path.join("assets", "tmp"),
        force_update=False,
    )
    observations = filter_observations(observations)

    logging.info("Calculating decisiontree completeness score for cavs...")
    completeness_score = get_completeness_score(
        probes=mock_probes,
        concepts=cav_names.copy(),
        model=model,
        observations=observations,
        layer_idx=-1,
        epochs=epochs,
        ignore_layers=ignore_layers,
        method="decisiontree",
        concept_score_method="soft",
        verbose=False,
        result_path=result_path,
    )

    cav_names.remove("random")

    mock_activations = {}
    for i, act in positive_activations.items():
        mock_activations[str(i)] = {"latest": {"layer": {"output": act}}}

    logging.info("Calculating statistics for cavs...")
    calculate_statistics(
        concepts=cav_names.copy(),
        activations=mock_activations,
        probes=mock_probes,
        layer_idx=-1,
        results_path=result_path,
        filename="cav_statistics.json",
    )
    calculate_probe_similarities(
        mock_probes, -1, result_path, filename="cav_similarity.json"
    )


if __name__ == "__main__":
    main()
