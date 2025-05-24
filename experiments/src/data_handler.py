from experiments.src.dataset_generator import generate_dataset
from experiments.src.file_handler import store_images_from_loader
import torch


def prepare_data(
    dataset_name,
    batch_size,
    test_batch_size,
    n_train: int | None = None,
    n_test: int | None = None,
):
    train_loader, test_loader = generate_dataset(
        name=dataset_name,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        n_train=n_train,
        n_test=n_test,
    )
    store_images_from_loader(train_loader)
    return train_loader, test_loader


def flatten_data_loader(loader):
    all_X = torch.cat([data for data, _ in loader], dim=0)
    all_targets = torch.cat([target for _, target in loader], dim=0)
    return all_X, all_targets
