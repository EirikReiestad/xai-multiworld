import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", choices=["probe", "tcav"], help="Specify the method to use"
)
args = parser.parse_args()

if args.method == "probe":
    subprocess.run(["python", "xailib/core/linear_probing"])
elif args.method == "tcav":
    pass
