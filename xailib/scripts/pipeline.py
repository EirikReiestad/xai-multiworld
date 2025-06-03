import json
import logging
import os
import time

from joblib.pool import np
from sklearn.linear_model import LogisticRegression

from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import download_models, get_latest_model, get_models
from utils.common.observation import filter_observations
from utils.core.model_loader import ModelLoader
from xailib.common.completeness_score import get_completeness_score
from xailib.common.generate_concepts import generate_concepts
from xailib.core.calculate_cavs.calculate_cavs import calculate_cavs
from xailib.core.shap.calculate_shap import calculate_shap
from xailib.utils.activations import get_activations, get_concept_activations
from xailib.utils.metrics import (
    calculate_cav_similarity,
    calculate_probe_robustness,
    calculate_probe_similarities,
    calculate_statistics,
    get_concept_scores,
    get_tcav_scores,
)
from xailib.utils.observation import get_observations
from xailib.utils.probes import get_probes_and_activations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    with open("xailib/configs/pipeline_config.json", "r") as f:
        config = json.load(f)

    current_time = time.strftime("%Y%m%d-%H%M%S")

    result_path = os.path.join("pipeline", f"{current_time}", "results")
    figure_path = os.path.join("pipeline", f"{current_time}", "figures")
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    logging.info("Downloading models...")
    download_models(
        low=config["wandb"]["models"]["low"],
        high=config["wandb"]["models"]["high"],
        step=config["wandb"]["models"]["step"],
        model_name=config["wandb"]["models"]["name"],
        wandb_project_folder=config["wandb"]["project_folder"],
        artifact_path=config["path"]["artifacts"],
        force_update=config["force_update"],
    )

    artifact = ModelLoader.load_latest_model_artifacts_from_path(
        config["path"]["artifacts"]
    )

    logging.info("Creating environment...")
    environment = create_environment(artifact)

    models = get_models(
        artifact=artifact,
        model_type=config["model"]["type"],
        env=environment,
        eval=True,
        artifact_path=config["path"]["artifacts"],
    )
    latest_model = get_latest_model(
        artifact=artifact,
        model_type=config["model"]["type"],
        artifact_path=config["path"]["artifacts"],
        env=environment,
        eval=True,
    )
    logging.info(latest_model)

    logging.info("Collecting rollouts...")
    observations = collect_rollouts(
        env=environment,
        artifact=artifact,
        n=config["collect_rollouts"]["observations"],
        method=config["model"]["method"],
        observation_path=config["path"]["observations"],
        force_update=config["force_update"],
        model_type=config["model"]["type"],
        sample_rate=config["collect_rollouts"]["sample_rate"],
        artifact_path=config["path"]["artifacts"],
    )
    observations = filter_observations(observations)

    logging.info("Generating concepts...")
    generate_concepts(
        concepts=config["concepts"],
        env=environment,
        observations=config["generate_concepts"]["observations"],
        artifact=artifact,
        method=config["model"]["method"],
        model_type=config["model"]["type"],
        force_update=config["force_update"],
        artifact_path=config["path"]["artifacts"],
        concept_path=config["path"]["concepts"],
        result_dir=result_path,
    )

    logging.info("Loading concept observations...")
    (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    ) = get_observations(config["concepts"])

    logging.info("Getting probes and activations for concepts...")
    probes, positive_activations, negative_activations = get_probes_and_activations(
        config["concepts"],
        config["analyze"]["ignore_layers"],
        models,
        positive_observations,
        negative_observations,
    )
    logging.info("Calculating network completeness score for concepts...")

    if config["completeness_score"]["method"] == "network":
        completeness_score = get_completeness_score(
            probes=probes,
            concepts=config["concepts"],
            model=latest_model,
            observations=observations,
            layer_idx=config["analyze"]["layer_idx"],
            epochs=config["completeness_score"]["network_epochs"],
            ignore_layers=config["analyze"]["ignore_layers"],
            method="network",
            concept_score_method=config["completeness_score"]["concept_score_method"],
            verbose=False,
            result_path=result_path,
            filename="concept_completeness_score_network.json",
        )
    logging.info("Calculating decisiontree completeness score for concepts...")
    completeness_score = get_completeness_score(
        probes=probes,
        concepts=config["concepts"],
        model=latest_model,
        observations=observations,
        layer_idx=config["analyze"]["layer_idx"],
        epochs=config["completeness_score"]["decisiontree_epochs"],
        ignore_layers=config["analyze"]["ignore_layers"],
        method="decisiontree",
        concept_score_method=config["completeness_score"]["concept_score_method"],
        verbose=False,
        result_path=result_path,
        filename="concept_completeness_score_decision_tree.json",
    )

    test_positive_activations, test_input, test_output = get_concept_activations(
        concepts=config["concepts"],
        observation=test_positive_observations,
        models=models,
        ignore_layers=config["analyze"]["ignore_layers"],
    )

    activations, input, output = get_activations(
        {"latest": latest_model}, observations, config["analyze"]["ignore_layers"]
    )
    logging.info("Calculating concept scores for concepts...")
    concept_scores = get_concept_scores(
        config["concepts"], test_positive_activations, probes, result_path, figure_path
    )
    logging.info("Calculating TCAV scores for concepts...")
    tcav_scores = get_tcav_scores(
        config["concepts"],
        test_positive_activations,
        test_output,
        probes,
        result_path,
        figure_path,
    )
    logging.info("Calculating probe robustness for concepts...")
    calculate_probe_robustness(
        concepts=config["concepts"],
        model=latest_model,
        splits=config["analyze"]["splits"],
        layer_idx=config["analyze"]["layer_idx"],
        epochs=config["analyze"]["robustness_epochs"],
        results_path=result_path,
    )
    logging.info("Calculating statistics for concepts...")
    calculate_statistics(
        concepts=config["concepts"],
        activations=positive_activations,
        probes=probes,
        layer_idx=config["analyze"]["layer_idx"],
        results_path=result_path,
    )
    calculate_probe_similarities(probes, config["analyze"]["layer_idx"], result_path)

    # This is just some estimation of the value K.
    # It is chosen from domain knowledge, but here we just estimate average_instance_of_each_class and total_number_of_instances
    average_instance_of_each_class = 100
    total_number_of_instances = 1000
    average_class_ratio = average_instance_of_each_class / total_number_of_instances
    K = int(config["calculate_cavs"]["batch_size"] * average_class_ratio / 2)

    logging.info("Calculating CAVs...")
    (
        cavs,
        stats,
        positive_cav_observations,
        negative_cav_observations,
        positive_cav_activations,
        negative_cav_activations,
        similarity_weights,
    ) = calculate_cavs(
        model=latest_model,
        env=environment,
        artifact=artifact,
        method=config["model"]["method"],
        M=config["calculate_cavs"]["M"],
        K=K,
        lambda_1=config["calculate_cavs"]["lambda_1"],
        lambda_2=config["calculate_cavs"]["lambda_2"],
        lambda_3=config["calculate_cavs"]["lambda_3"],
        batch_size=config["calculate_cavs"]["batch_size"],
        lr=config["calculate_cavs"]["lr"],
        epochs=config["calculate_cavs"]["epochs"],
        convergence_threshold=config["calculate_cavs"]["convergence_threshold"],
        result_path=result_path,
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

    mock_probes = {}
    for i, cav in cavs.items():
        mock_probe = LogisticRegression()
        mock_probe.coef_ = cav
        mock_probe.intercept_ = 0
        mock_probe.classes_ = [0, 1]
        mock_probes[str(i)] = {"latest": {"layer": mock_probe}}

    cav_names = list(mock_probes.keys())

    calculate_cav_similarity(cavs, result_path, "cav_similarity.json")

    logging.info("Calculating network completeness score for cavs...")
    if (
        config["completeness_score"]["method"] == "network"
        and config["calculate_cavs"]["M"] < 8
    ):
        completeness_score = get_completeness_score(
            probes=mock_probes,
            concepts=cav_names,
            model=latest_model,
            observations=observations,
            layer_idx=-1,
            epochs=config["completeness_score"]["network_epochs"],
            ignore_layers=config["analyze"]["ignore_layers"],
            method="network",
            concept_score_method="soft",
            verbose=False,
            result_path=result_path,
            filename="cav_completeness_score_network.json",
        )
    logging.info("Calculating decisiontree completeness score for cavs...")
    completeness_score = get_completeness_score(
        probes=mock_probes,
        concepts=cav_names.copy(),
        model=latest_model,
        observations=observations,
        layer_idx=-1,
        epochs=config["completeness_score"]["decisiontree_epochs"],
        ignore_layers=config["analyze"]["ignore_layers"],
        method="decisiontree",
        concept_score_method="soft",
        verbose=False,
        result_path=result_path,
        filename="cav_completeness_score_decision_tree.json",
    )

    mock_activations = {}
    for i, act in positive_cav_activations.items():
        mock_activations[str(i)] = {"latest": {"layer": {"output": act}}}

    logging.info("Calculating statistics for cavs...")
    calculate_statistics(
        concepts=cav_names,
        activations=mock_activations,
        probes=mock_probes,
        layer_idx=-1,
        results_path=result_path,
        filename="cav_statistics.json",
    )
    calculate_probe_similarities(
        mock_probes, -1, result_path, filename="cav_similarity.json"
    )

    # JOINTLY RESULTS
    concept_names = config["concepts"] + cav_names
    mock_activations.update(positive_activations)
    mock_probes.update(probes)

    logging.info("Calculating statistics for concepts and cavs...")
    calculate_statistics(
        concepts=concept_names,
        activations=mock_activations,
        probes=mock_probes,
        layer_idx=-1,
        results_path=result_path,
        filename="concept_cav_statistics.json",
    )
    calculate_probe_similarities(
        mock_probes, -1, result_path, filename="concept_cav_similarity.json"
    )

    logging.info("Calculating SHAP")
    calculate_shap(
        artifact=artifact,
        environment=environment,
        model=latest_model,
        observations=observations,
        save_path=result_path,
    )


if __name__ == "__main__":
    main()
