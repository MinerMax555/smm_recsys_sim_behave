import subprocess
import sys


def call_script(n, dataset="example", model="ItemKNN", choice_model="rank_based"):
    for i in range(1, n + 1):
        command = [
            sys.executable, "main.py", dataset, str(i),
            "--clean",
            "--model", model,
            "--choice-model", choice_model
        ]
        result = subprocess.run(command, check=True)

        if result.returncode == 0:
            print(f"Iteration {i}: Success")
        else:
            print(f"Iteration {i}: Error")


if __name__ == "__main__":
    k = 100
    dataset = "example"
    model = "ItemKNN"
    choice_model = "rank_based"
    call_script(k, dataset, model, choice_model)
