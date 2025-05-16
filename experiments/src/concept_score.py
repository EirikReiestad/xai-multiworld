import torch
from src.dataset_generator import generate_concept_data
from src.network import Net
from src.utils import (
    get_concept_activations,
    get_concept_scores,
    get_probes_and_activations,
    get_tcav_scores,
    store_images,
)


def main():
    name = "square"
    model_name = name + ".pt"
    model = Net().to()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    ) = generate_concept_data(name, num_samples=100)
    for key in positive_observations.keys():
        store_images(
            torch.tensor(positive_observations[key][0:20]),
            f"tmp/positive_observations/{key}",
        )
        store_images(
            torch.tensor(negative_observations[key][0:20]),
            f"tmp/negative_observations/{key}",
        )
        store_images(
            torch.tensor(test_positive_observations[key][0:20]),
            f"tmp/test_positive_observations/{key}",
        )
        store_images(
            torch.tensor(test_negative_observations[key][0:20]),
            f"tmp/test_negative_observations/{key}",
        )

    models = {"latest": model}
    ignore_layers = []
    test_positive_activations, test_input, test_output = get_concept_activations(
        test_positive_observations, models, ignore_layers
    )
    probes, positive_activations, negative_activations = get_probes_and_activations(
        ignore_layers=ignore_layers,
        models=models,
        positive_observations=positive_observations,
        negative_observations=negative_observations,
    )
    concept_scores = get_concept_scores(
        list(test_positive_activations.keys()),
        test_positive_activations,
        probes,
        show=False,
    )
    tcav_scores = get_tcav_scores(
        list(test_positive_activations.keys()),
        test_positive_activations,
        test_output,
        probes,
        show=False,
    )


if __name__ == "__main__":
    main()
