import os
import json
import ast

import pandas as pd
from torchvision.utils import save_image


def read_multi_files(filename: str, n: int, result_dir: str = "experiments/results"):
    dfs = []
    for i in range(n):
        df = pd.read_json(os.path.join(result_dir, f"{filename}_{i}.json")).T
        dfs.append(df)
    df = pd.concat(dfs)
    df.index = range(len(df))
    return df


def store_images_from_loader(loader, folder="experiments/tmp", num_images=10):
    os.makedirs(folder, exist_ok=True)
    count = 0
    for images, labels in loader:
        for i in range(images.size(0)):
            if count >= num_images:
                return
            save_image(
                images[i], os.path.join(folder, f"image_{count + 1}_{labels[i]}.png")
            )
            count += 1


def read_results(path: str) -> dict:
    with open(path, "r") as f:
        results = json.load(f)
        results = {ast.literal_eval(key): value for key, value in results.items()}
    return results


def write_results(results: dict, path: str, custom_cls=None):
    with open(path, "w") as f:
        results = {str(key): value for key, value in results.items()}
        json.dump(results, f, cls=custom_cls)


def store_images(images, folder="experiments/tmp"):
    os.makedirs(folder, exist_ok=True)
    for i, image in enumerate(images):
        try:
            save_image(image, os.path.join(folder, f"image_{i}.png"))
        except IndexError as e:
            print(image)
            raise e
