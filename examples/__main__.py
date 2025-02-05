import faulthandler
import subprocess

faulthandler.enable(file=open("crash_log.txt", "w"))

if __name__ == "__main__":
    script = "examples/multi_flock.py"
    subprocess.run(["python", script])
