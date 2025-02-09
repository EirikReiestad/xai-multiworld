import argparse
import logging
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cs",
        "--concept-score",
        action="store_true",
        help="Run the concept score script",
    )
    parser.add_argument(
        "-ts", "--tcav-score", action="store_true", help="Run the tcav score script"
    )
    parser.add_argument(
        "-cb",
        "--concept-backprop",
        action="store_true",
        help="Run the concept backpropagation script",
    )
    parser.add_argument("--shap", action="store_true", help="Run the shap script")

    args, unknown = parser.parse_known_args()

    concept_score_subprocess_args = ["python", "xailib/scripts/concept_score.py"]
    tcav_score_subprocess_args = ["python", "xailib/scripts/tcav.py"]
    concept_backprop_subprocess_args = [
        "python",
        "xailib/scripts/concept_backpropagation.py",
    ]
    shap_subprocess_args = ["python", "xailib/scripts/shap_score.py"]

    if args.concept_score:
        concept_score_subprocess_args += ["--concept-score"]
    if args.tcav_score:
        tcav_score_subprocess_args += ["--tcav-score"]
    if args.concept_backprop:
        concept_backprop_subprocess_args += ["--concept-backprop"]
    if args.shap:
        shap_subprocess_args += ["--shap"]

    for subprocess_args in [
        concept_score_subprocess_args,
        tcav_score_subprocess_args,
        concept_backprop_subprocess_args,
        shap_subprocess_args,
    ]:
        if len(subprocess_args) > 2:
            subprocess.run(subprocess_args)


if __name__ == "__main__":
    main()
