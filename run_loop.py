import subprocess
import sys


def call_script(n):
    for i in range(1, n+1):
        command = [sys.executable, "main.py", "example", str(i), "--clean", "--choice-model", "rank_based"]
        result = subprocess.run(command, check=True)

        if result.returncode == 0:
            print(f"Iteration {i}: Success")
        else:
            print(f"Iteration {i}: Error")


if __name__ == "__main__":
    k = 10
    call_script(k)
