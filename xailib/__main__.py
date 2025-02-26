import argparse
import logging
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline", action="store_true", help="Run the pipeline script"
    )
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
        "-po",
        "--probe-observation",
        action="store_true",
        help="Run the probe observation script",
    )
    parser.add_argument(
        "-cb",
        "--concept-backprop",
        action="store_true",
        help="Run the concept backpropagation script",
    )
    parser.add_argument(
        "-csm",
        "--completeness-score",
        action="store_true",
        help="Run the completeness score statistic script",
    )
    parser.add_argument(
        "-ps",
        "--probe-statistic",
        action="store_true",
        help="Run the probe statistic script",
    )
    parser.add_argument(
        "-pr",
        "--probe-robustness",
        action="store_true",
        help="Run the probe robustness script",
    )
    parser.add_argument("--shap", action="store_true", help="Run the shap script")

    args, unknown = parser.parse_known_args()

    pipeline_subprocess_args = ["python", "xailib/scripts/pipeline.py"]
    concept_score_subprocess_args = ["python", "xailib/scripts/concept_score.py"]
    tcav_score_subprocess_args = ["python", "xailib/scripts/tcav_score.py"]
    concept_backprop_subprocess_args = [
        "python",
        "xailib/scripts/concept_backpropagation.py",
    ]
    probe_observation_subprocess_args = [
        "python",
        "xailib/scripts/probe_observation.py",
    ]
    completeness_score_subprocess_args = [
        "python",
        "xailib/scripts/completeness_score.py",
    ]
    probe_statistic_subprocess_args = ["python", "xailib/scripts/probe_statistics.py"]
    probe_robustness_subprocess_args = ["python", "xailib/scripts/probe_robustness.py"]
    shap_subprocess_args = ["python", "xailib/scripts/shap_score.py"]

    if args.pipeline:
        pipeline_subprocess_args += ["--pipeline"]
    if args.concept_score:
        concept_score_subprocess_args += ["--concept-score"]
    if args.tcav_score:
        tcav_score_subprocess_args += ["--tcav-score"]
    if args.probe_observation:
        probe_observation_subprocess_args += ["--probe-observation"]
    if args.concept_backprop:
        concept_backprop_subprocess_args += ["--concept-backprop"]
    if args.completeness_score:
        completeness_score_subprocess_args += ["--completeness-score"]
    if args.probe_statistic:
        probe_statistic_subprocess_args += ["--probe-statistic"]
    if args.probe_robustness:
        probe_robustness_subprocess_args += ["--probe-robustness"]
    if args.shap:
        shap_subprocess_args += ["--shap"]

    for subprocess_args in [
        pipeline_subprocess_args,
        concept_score_subprocess_args,
        tcav_score_subprocess_args,
        probe_observation_subprocess_args,
        concept_backprop_subprocess_args,
        completeness_score_subprocess_args,
        probe_statistic_subprocess_args,
        probe_robustness_subprocess_args,
        shap_subprocess_args,
    ]:
        if len(subprocess_args) > 2:
            subprocess.run(subprocess_args)


if __name__ == "__main__":
    main()
