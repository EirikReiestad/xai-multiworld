from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def generate_dataset(
    name: Literal["mnist", "square"], batch_size: int, test_batch_size: int
):
    name = name.lower()
    if name == "mnist":
        return generate_mnist_dataset(
            batch_size=batch_size, test_batch_size=test_batch_size
        )
    elif name == "square":
        return generate_square_dataset(
            batch_size=batch_size, test_batch_size=test_batch_size
        )
    else:
        raise ValueError(f"name {name} is not supported.")


# Function to generate non-overlapping square data
def generate_square_data(
    num_samples: int,
    img_size: int,
    num_squares: int | None = None,
    exclude: list[int] = [],
    max_squares: int = 6,
):
    data = []
    labels = []
    choices = [*range(0, max_squares)] if num_squares is None else [num_squares]
    choices = [x for x in choices if x not in exclude]
    assert all([isinstance(x, int) for x in choices])
    if choices == [] or choices is None:
        raise ValueError("no choices:(")
    for _ in range(num_samples):
        squares = num_squares or np.random.choice(choices)
        if squares is None:
            raise ValueError("noooo, an error")
        img = np.zeros((img_size, img_size))
        positions = []
        for _ in range(squares):
            while True:
                x, y = np.random.randint(0, img_size - 5, size=2)
                # size = np.random.randint(2, 5)
                size = 3
                square = (x, y, size)
                if all(
                    (
                        x + size <= p[0]
                        or x >= p[0] + p[2]
                        or y + size <= p[1]
                        or y >= p[1] + p[2]
                    )
                    for p in positions
                ):
                    positions.append(square)
                    img[x : x + size, y : y + size] = 1
                    break
        data.append(img)
        labels.append(squares)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)


def generate_square_dataset(
    img_size: int = 28,
    train_samples: int = 8000,
    test_samples: int = 2000,
    batch_size: int = 32,
    test_batch_size: int = 1000,
    max_squares: int = 9,
):
    X_train, y_train = generate_square_data(
        train_samples, img_size, max_squares=max_squares, exclude=[4, 5, 8]
    )
    X_test, y_test = generate_square_data(
        test_samples, img_size, max_squares=max_squares, exclude=[4, 5, 8]
    )

    # Convert to tensors
    X_train = torch.tensor(X_train).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create datasets
    dataset1 = TensorDataset(X_train, y_train)
    dataset2 = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset2, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def generate_mnist_dataset(batch_size: int = 32, test_batch_size: int = 32):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=test_batch_size, shuffle=False
    )
    return train_loader, test_loader


def generate_concept_data(model_name: Literal["square"], num_samples: int, img_size=28):
    if model_name == "square":
        return generate_square_concept_data(num_samples=num_samples, img_size=img_size)


def generate_square_concept_data(
    num_samples: int = 1000, img_size=28, max_squares: int = 9
):
    positive_observations = {}
    negative_observations = {}
    test_positive_observations = {}
    test_negative_observations = {}
    for i in range(0, max_squares):
        x, y = generate_square_data(
            num_samples=num_samples,
            img_size=img_size,
            num_squares=i,
        )
        x_test, y_test = generate_square_data(
            num_samples=num_samples, img_size=img_size, num_squares=i
        )
        x_neg, y_neg = generate_square_data(
            num_samples=num_samples,
            img_size=img_size,
            exclude=[i],
            max_squares=max_squares,
        )
        x_neg_test, y_neg_test = generate_square_data(
            num_samples=num_samples,
            img_size=img_size,
            exclude=[i],
            max_squares=max_squares,
        )
        positive_observations[i] = x
        negative_observations[i] = x_neg
        test_positive_observations[i] = x_test
        test_negative_observations[i] = x_neg_test

    return (
        positive_observations,
        negative_observations,
        test_positive_observations,
        test_negative_observations,
    )
