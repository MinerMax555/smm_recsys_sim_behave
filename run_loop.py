import subprocess
import sys

import argh
from argh import arg


@arg('-n', type=int, help='Number of iterations to run')
@arg('--dataset', type=str, help='Name of the dataset (a subfolder under experiments/) to be evaluated')
@arg('--model', type=str, help='Name of RecBole model to be used')
@arg('--choice-model', type=str, help='Name of choice model to be used.')
def call_script(n=100, dataset="example", model="ItemKNN", choice_model="rank_based"):
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
    argh.dispatch_command(call_script)