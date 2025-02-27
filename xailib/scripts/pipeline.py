import json
import logging

from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import download_models, get_latest_model, get_models
from utils.core.model_loader import ModelLoader
from xailib.common.completeness_score import get_completeness_score
from xailib.common.generate_concepts import generate_concepts
from xailib.utils.activations import get_activations, get_concept_activations
from xailib.utils.metrics import (
    calculate_probe_robustness,
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

    logging.info("Downloading models...")
    download_models(
        low=config["wandb"]["models"]["low"],
        high=config["wandb"]["models"]["high"],
        step=config["wandb"]["models"]["step"],
        wandb_project_folder=config["wandb"]["project_folder"],
        artifact_path=config["path"]["artifacts"],
        force_update=config["force_update"],
    )

    artifact = ModelLoader.load_latest_model_artifacts_from_path(
        config["path"]["artifacts"]
    )

    logging.info("Creating environment...")
    environment = create_environment(artifact)
    logging.info("Collecting rollouts...")
    observations = collect_rollouts(
        env=environment,
        artifact=artifact,
        n=config["collect_rollouts"]["observations"],
        method=config["collect_rollouts"]["method"],
        observation_path=config["path"]["observations"],
        force_update=config["force_update"],
        model_type=config["model"]["type"],
        sample_rate=config["collect_rollouts"]["sample_rate"],
        artifact_path=config["path"]["artifacts"],
    )
    logging.info("Generating concepts...")
    generate_concepts(
        concepts=config["concepts"],
        env=environment,
        observations=config["generate_concepts"]["observations"],
        artifact=artifact,
        method=config["generate_concepts"]["method"],
        model_type=config["model"]["type"],
        force_update=config["force_update"],
        artifact_path=config["path"]["artifacts"],
        observation_path=config["path"]["observations"],
    )

    logging.info("Loading observations...")
    (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    ) = get_observations(config["concepts"])

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

    logging.info("Getting probes and activations...")
    probes, positive_activations, negative_activations = get_probes_and_activations(
        config["concepts"],
        config["analyze"]["ignore_layers"],
        models,
        positive_observations,
        negative_observations,
    )
    logging.info("Calculating completeness score...")
    completeness_score = get_completeness_score(
        probes=probes,
        concepts=config["concepts"],
        artifact=artifact,
        environment=environment,
        observations=observations,
        layer_idx=config["analyze"]["layer_idx"],
        ignore_layers=config["analyze"]["ignore_layers"],
        model_type=config["model"]["type"],
        method=config["completeness_score"]["method"],
        artifact_path=config["path"]["artifacts"],
        eval=True,
        verbose=False,
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
    logging.info("Calculating concept scores...")
    concept_scores = get_concept_scores(
        config["concepts"], test_positive_activations, probes, config["path"]["results"]
    )
    logging.info("Calculating TCAV scores...")
    tcav_scores = get_tcav_scores(
        config["concepts"],
        test_positive_activations,
        test_output,
        probes,
        config["path"]["results"],
    )
    logging.info("Calculating probe robustness...")
    calculate_probe_robustness(
        concepts=config["concepts"],
        model=latest_model,
        splits=config["analyze"]["splits"],
        layer_idx=config["analyze"]["layer_idx"],
        epochs=config["analyze"]["robustness_epochs"],
        results_path=config["path"]["results"],
    )
    logging.info("Calculating statistics...")
    calculate_statistics(
        concepts=config["concepts"],
        activations=positive_activations,
        layer_idx=config["analyze"]["layer_idx"],
        results_path=config["path"]["results"],
    )


if __name__ == "__main__":
    main()
