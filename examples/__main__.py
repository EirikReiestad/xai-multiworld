import faulthandler
import subprocess

faulthandler.enable(file=open("crash_log.txt", "w"))

if __name__ == "__main__":
    script = "examples/go_to_goal.py"
    subprocess.run(["python", script])
