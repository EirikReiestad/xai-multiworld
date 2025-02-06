import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()

    args, unknown = parser.parse_known_args()

    subprocess.run(["python", "utils"] + unknown)

    subprocess.run(["python", "xailib"] + unknown)


if __name__ == "__main__":
    main()
