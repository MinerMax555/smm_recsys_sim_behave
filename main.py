import argparse
import shutil
from pathlib import Path

import argh
from argh import arg
from recbole.config import Config
from recbole.quick_start import run_recbole

from recbole_wrapper import run_recbole_experiment

EXPERIMENTS_FOLDER = Path('experiments')


def prepare_run(dataset_name: str, iteration: int, clean=False) -> tuple[Path, Path, Path]:
    """
    Prepares dataset folder structure and asserts all required files are present.
    :param dataset_name: name of the dataset to be used
    :param iteration: iteration number
    :param clean: if True, deletes all files in the data/ output/ and logs/ folders that may be present
                  Only has an effect if iteration number is 1
    :returns: a tuple of paths containing:
      - interactions file for the correct iteration
      - demographics file
      - tracks file
    """
    # Python/Pathlib magic: the division operator on Path objects functions as os.path.join()
    experiment_folder = EXPERIMENTS_FOLDER / dataset_name
    if not experiment_folder.exists():
        raise FileNotFoundError(f'Dataset {dataset_name} not found')

    # Assert that the required files are present in input/
    if not (experiment_folder / 'input').exists():
        raise FileNotFoundError(f'Dataset invalid: Input folder missing')

    demographics_file = experiment_folder / 'input' / 'demographics.tsv'
    if not demographics_file.exists():
        raise FileNotFoundError(f'Dataset invalid: input/demographics.tsv file missing')

    tracks_file = experiment_folder / 'input' / 'tracks.tsv'
    if not tracks_file.exists():
        raise FileNotFoundError(f'Dataset invalid: input/tracks.tsv file missing')

    # Create recbole_workdir if it doesn't exist
    (Path('recbole_tmp') / 'dataset').mkdir(exist_ok=True, parents=True)

    if iteration == 1:
        input_inter_file = experiment_folder / 'input' / f'dataset.inter'
        if not input_inter_file.exists():
            raise FileNotFoundError(f'Dataset invalid: input/dataset.inter')

        if clean:
            shutil.rmtree(experiment_folder / 'datasets', ignore_errors=True)
            shutil.rmtree(experiment_folder / 'output', ignore_errors=True)
            shutil.rmtree(experiment_folder / 'log', ignore_errors=True)
            shutil.rmtree(experiment_folder / 'log_tensorboard', ignore_errors=True)
        (experiment_folder / 'datasets').mkdir(exist_ok=True)
        (experiment_folder / 'output').mkdir(exist_ok=True)
        (experiment_folder / 'log').mkdir(exist_ok=True)
        (experiment_folder / 'log_tensorboard').mkdir(exist_ok=True)

        # Copy the interactions file to datasets/iteration_1.inter
        shutil.copy(input_inter_file, experiment_folder / 'datasets' / f'iteration_{iteration}.inter')

    inter_file = experiment_folder / f'datasets' / f'iteration_{iteration}.inter'

    # Copy inter file into recbole_workdir/iteration_{iteration}/dataset.inter
    shutil.copy(inter_file, Path('recbole_tmp') / f'dataset' / 'dataset.inter')

    return inter_file, demographics_file, tracks_file


def cleanup(dataset_name: str, iteration: int):
    """Cleanup after iteration is complete"""
    # create subfolders for logs
    log_folder = EXPERIMENTS_FOLDER / dataset_name / 'log' / f'iteration_{iteration}'
    shutil.rmtree(log_folder, ignore_errors=True)
    log_folder.mkdir(exist_ok=True)
    # move any logs that were written into this folder
    for log_file in Path('log').iterdir():
        shutil.move(log_file, log_folder)
    # move anything in log_tensorbpard to the correct folder
    for log_file in Path('log_tensorboard').iterdir():
        shutil.move(log_file, log_folder)


    # remove the saved, log, log_tensorboard and recbole_workdir folder
    shutil.rmtree('saved', ignore_errors=True)
    shutil.rmtree('log', ignore_errors=True)
    shutil.rmtree('log_tensorboard', ignore_errors=True)
    shutil.rmtree('recbole_tmp', ignore_errors=True)


@arg('dataset_name', type=str, help='Name of the dataset (a subfolder under data/) to be evaluated')
@arg('iteration', type=int,
     help='Iteration number. Recbole training will use data from dataset_<iteration-1> and output to dataset_<iteration>')
@arg('-m', '--model', type=str, help='Name of RecBole model to be used')
@arg('-cm', '--choice-model', type=str, help='Name of choice model to be used. See README for available options')
@arg('-c', '--config', type=str, help='Path to the Recbole config file')
@arg('-k', type=int, help='Number of items to be recommended per user')
@arg('--clean', action=argparse.BooleanOptionalAction,
     help='If True, deletes all files in the data/ output/ and logs/ folders that may be present')
def do_single_loop(
        dataset_name, iteration, model='ItemKNN', choice_model='random',
        config='recbole_config_default.yaml',
        k=10, clean=False):
    """
    Executes a single iteration loop consisitng of training, evaluation and the
    addition of new interactions by a choice model. This file only does a single loop at a time and needs to be called as a subprocess.
    This is due to weirdness (coming from Recbole?) leading to a memory leak and swiftly decreasing performance after a few iterations.
    """
    print(f'Preparing iteration {iteration} for dataset/experiment {dataset_name}...')
    data_path, demographics_path, tracks_path = prepare_run(dataset_name, iteration, clean)
    print(' Done!')

    trained_model = run_recbole_experiment(model=model, dataset=dataset_name, iteration=iteration, config_path=config)

    # Cleanup after finished loop
    cleanup(dataset_name, iteration)


if __name__ == '__main__':
    argh.dispatch_command(do_single_loop)
