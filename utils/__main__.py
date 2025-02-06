import argparse
import logging
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dms",
        "--download-models",
        nargs=4,
        metavar=("project_folder", "low", "high", "step"),
        help="Download models with optional arguments: [project_folder] [low] [high] [step]",
    )
    parser.add_argument(
        "-dm",
        "--download-model",
        nargs="*",
        metavar=("model_name", "version"),
        help="Download model with optional arguments: [model_name] [version]",
    )
    parser.add_argument(
        "-gc",
        "--generate-concepts",
        nargs="*",
        metavar=("n"),
        help="Generate n concepts dataset (for all possible concepts)",
    )

    args, unknown = parser.parse_known_args()

    model_downloader_subprocess_args = ["python", "utils/scripts/model_downloader.py"]
    generate_concepts_subprocess_args = ["python", "utils/scripts/generate_concepts.py"]

    if args.download_models is not None:
        logging.info(f"Arguments for --download-models: {args.download_models}")
        model_downloader_subprocess_args += ["--download-models"] + args.download_models

    if args.download_model is not None:
        logging.info(f"Arguments for --download-model: {args.download_model}")
        model_downloader_subprocess_args += ["--download-model"] + args.download_model
    if args.generate_concepts is not None:
        logging.info(f"Arguments for --generate-concepts: {args.generate_concepts}")
        generate_concepts_subprocess_args += [
            "--generate-concepts"
        ] + args.generate_concepts

    if len(model_downloader_subprocess_args) > 2:
        subprocess.run(model_downloader_subprocess_args)
    if len(generate_concepts_subprocess_args) > 2:
        subprocess.run(generate_concepts_subprocess_args)


if __name__ == "__main__":
    main()
