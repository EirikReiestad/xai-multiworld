import logging
import os

from utils.common.collect_rollouts import collect_rollouts
from utils.common.environment import create_environment
from utils.common.model import get_models
from utils.common.numpy_collections import NumpyEncoder
from utils.common.write import write_results
from utils.core.model_loader import ModelLoader
from xailib.core.calculate_cavs.calculate_cavs import calculate_cavs
from xailib.utils.metrics import calculate_cav_similarity, calculate_probe_similarities
from xailib.utils.observation import get_observations
from xailib.utils.probes import get_probes_and_activations


def main():
    artifact_path = "artifacts"
    model_type = "dqn"
    eval = True
    ignore_layers = ["_fc0"]
    result_path = os.path.join("assets", "results")

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact, static=False)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=eval,
        artifact_path=artifact_path,
    )
    model = list(models.values())[-1]

    M = 100
    lambda_1 = 0.1
    lambda_2 = 0.1
    batch_size = 128
    lr = 1e-3
    epochs = 10

    average_instance_of_each_class = 100
    total_number_of_instances = 1000
    average_class_ratio = average_instance_of_each_class / total_number_of_instances

    K = int(batch_size * average_class_ratio / 2)

    cavs, stats = calculate_cavs(
        model=model,
        env=environment,
        artifact=artifact,
        method="policy",
        M=M,
        K=K,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        ignore_layers=ignore_layers,
    )

    cavs = cavs.detach().cpu().numpy()
    results = {"cavs": cavs}
    write_results(
        results=results,
        path=os.path.join(result_path, "cavs.json"),
        custom_cls=NumpyEncoder,
    )

    cavs = {f"{i}": cav for i, cav in enumerate(cavs)}
    calculate_cav_similarity(cavs, result_path, "cav_similarity.json")

    # CALCULATE THE DEFINED CAVS

    artifact_path = "artifacts"
    model_type = "dqn"
    eval = True
    layer_idx = 4
    concepts = [
        "random",
        "goal_in_front",
        "goal_in_view",
        "goal_to_left",
        "goal_to_right",
        "wall_in_view",
    ]
    ignore_layers = ["_fc0"]

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
    models = get_models(
        artifact=artifact,
        model_type=model_type,
        env=environment,
        eval=eval,
        artifact_path=artifact_path,
    )
    (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    ) = get_observations(concepts)
    logging.info("Getting probes and activations...")
    probes, positive_activations, negative_activations = get_probes_and_activations(
        concepts=concepts,
        models=models,
        positive_observations=positive_observations,
        negative_observations=negative_observations,
        ignore_layers=ignore_layers,
    )
    logging.info("Calculating statistics...")
    calculate_probe_similarities(probes, layer_idx)
    defined_cavs = {}
    for key, value in probes.items():
        probe = list(list(value.values())[-1].values())[layer_idx]
        defined_cavs[key] = probe.coef_.flatten()
    cavs.update(defined_cavs)
    calculate_cav_similarity(cavs, result_path, filename="cav_similarities.json")


if __name__ == "__main__":
    main()
