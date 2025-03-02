from utils.common.environment import create_environment
from utils.common.model import get_latest_model
from utils.core.model_loader import ModelLoader
from xailib.utils.metrics import calculate_probe_robustness


def main():
    artifact_path = "artifacts"
    model_type = "dqn"
    eval = True
    concepts = [
        "random",
        "goal_in_front",
        "goal_in_view",
        "goal_to_left",
        "goal_to_right",
        "wall_in_view",
    ]
    ignore_layers = ["_fc0"]
    layer_idx = 4
    splits = [0.9, 0.5, 0.1]
    epochs = 10

    artifact = ModelLoader.load_latest_model_artifacts_from_path(artifact_path)
    environment = create_environment(artifact)
    latest_model = get_latest_model(
        artifact, model_type, artifact_path, environment, eval=eval
    )

    calculate_probe_robustness(
        concepts=concepts,
        model=latest_model,
        splits=splits,
        layer_idx=layer_idx,
        epochs=epochs,
        ignore_layers=ignore_layers,
    )


if __name__ == "__main__":
    main()
