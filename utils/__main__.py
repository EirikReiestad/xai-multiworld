import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    "--download-models",
    help="Download models from wandb, and store them in the artifacts folder",
    action="store_true",
)

args = parser.parse_args()

if args.download_models:
    subprocess.run(["python", "-m", "utils.scripts.model_downloader"])
