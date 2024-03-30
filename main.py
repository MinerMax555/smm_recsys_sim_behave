#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import argh
import numpy as np
import pandas as pd
from argh import arg
from recbole.config import Config
from recbole.quick_start import load_data_and_model
from tqdm import tqdm

from choice_models import accept_new_recommendations, prefilter_recommendations
from recbole_wrapper import run_recbole_experiment, get_recbole_scores

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

    shutil.rmtree('recbole_tmp', ignore_errors=True)
    shutil.rmtree('saved', ignore_errors=True)
    # Create recbole_workdir and saved if it doesn't exist
    (Path('recbole_tmp') / 'dataset').mkdir(exist_ok=True, parents=True)
    (Path('saved')).mkdir(exist_ok=True, parents=True)

    if iteration == 1:
        input_inter_file = experiment_folder / 'input' / f'dataset.inter'
        if not input_inter_file.exists():
            raise FileNotFoundError(f'Dataset invalid: input/dataset.inter')

        if clean:
            shutil.rmtree(experiment_folder / 'datasets', ignore_errors=True)
            shutil.rmtree(experiment_folder / 'output', ignore_errors=True)
            shutil.rmtree(experiment_folder / 'log', ignore_errors=True)
        (experiment_folder / 'datasets').mkdir(exist_ok=True)
        (experiment_folder / 'output').mkdir(exist_ok=True)
        (experiment_folder / 'log').mkdir(exist_ok=True)

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


def compute_top_k_scores(scores, dataset: str, iteration: int, k=10):
    """
    Computes the top k scores per user, saves it into the output folder
    and returns a dataframe with the new interactions
    """
    n = len(scores)
    item_ids, score = [], []

    for s in tqdm(scores, desc=f'calculating top-{k} item_ids', smoothing=0):
        items = np.argsort(-s)[:k]
        # Martin: Only get country-specific items:

        for item in items:
            item_ids.append(item)
            score.append(s[item])

    user_ids = []
    for i in range(n):
        user_ids.extend([i] * k)

    df = pd.DataFrame.from_dict({
        'user_id': user_ids,
        'item_id': item_ids,
        'rank': list(range(1, k + 1)) * n,
        'score': score
    })
    return df


@arg('dataset_name', type=str, help='Name of the dataset (a subfolder under data/) to be evaluated')
@arg('iteration', type=int,
     help='Iteration number. Recbole training will use data from dataset_<iteration-1> and output to dataset_<iteration>')
@arg('-m', '--model', type=str, help='Name of RecBole model to be used')
@arg('-cm', '--choice-model', type=str, help='Name of choice model to be used. See README for available options')
@arg('-c', '--config', type=str, help='Path to the Recbole config file')
@arg('-k', type=int, help='Number of items to be recommended per user')
@arg('-cc', '--control-country', type=str,
     help='Country to be used as a frozen control group that doesn\'t receive new recommendations')
@arg('--clean', action=argparse.BooleanOptionalAction,
     help='If True, deletes all files in the data/ output/ and logs/ folders that may be present')
def do_single_loop(
        dataset_name, iteration, model='ItemKNN', choice_model='random',
        config='recbole_config_default.yaml',
        k=10, control_country=None,
        clean=False):
    """
    Executes a single iteration loop consisitng of training, evaluation and the
    addition of new interactions by a choice model. This file only does a single loop at a time and needs to be called as a subprocess.
    This is due to weirdness (coming from Recbole?) leading to a memory leak and swiftly decreasing performance after a few iterations.
    """
    print(f'Preparing iteration {iteration} for dataset/experiment {dataset_name}...')
    data_path, demographics_path, tracks_path = prepare_run(dataset_name, iteration, clean)
    dataset_df = pd.read_csv(data_path, sep='\t', header=1, names=['user_id:token', 'item_id:token'])
    demographics_df = pd.read_csv(demographics_path, sep='\t', header=None,
                                  names=['country', 'age', 'gender', 'registration_time'])
    tracks_df = pd.read_csv(tracks_path, sep='\t', header=None, names=['title', 'artist', 'country'])
    print(' Done!')

    config = Config(model=model, dataset='dataset', config_file_list=[config])
    # Use Recbole to obtain a trained model
    run_recbole_experiment(model=model, dataset=dataset_name, iteration=iteration, config=config)

    # There should only be one model file in saved folder, get its path
    model_path = str(next(Path('saved').iterdir()))
    config, model, dataset, _, _, test_data = load_data_and_model(model_file=model_path)

    # Obtain recommendation scores
    scores = get_recbole_scores(model, dataset, test_data, config)

    # Obtain top k scores and save them for later analysis
    top_k_df = compute_top_k_scores(scores, dataset_name, iteration, k=k)
    top_k_df.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / 'output' / f'iteration_{iteration}_top_k.tsv', header=True,
        sep='\t', index=False)

    # Apply prefilters that remove invalid recommendations (such as for the control group)
    filtered_recs = prefilter_recommendations(top_k_df, demographics_df, tracks_df,
                                              control_country=control_country)

    # Apply Choice model to select new recommendations from the top k
    accepted_df = accept_new_recommendations(choice_model, filtered_recs, demographics_df, tracks_df)
    # Rename the columns to the correct names for recbole
    accepted_df = accepted_df[['user_id', 'item_id']].rename(
        columns={'user_id': 'user_id:token', 'item_id': 'item_id:token'})
    accepted_df.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / 'output' / f'iteration_{iteration}_accepted_songs.tsv',
        header=True, sep='\t', index=False)

    # Append new interactions to dataset_df
    df_new = pd.concat([dataset_df.copy(), accepted_df], ignore_index=True)
    # Sort them again such that all interactions are grouped by user_id
    df_new = df_new.sort_values(['user_id:token', 'item_id:token']).reset_index(drop=True)
    df_new.to_csv(
        EXPERIMENTS_FOLDER / dataset_name / 'datasets' / f'iteration_{iteration + 1}.inter',
        header=True, sep='\t', index=False)

    # Cleanup after finished loop
    cleanup(dataset_name, iteration)


if __name__ == '__main__':
    argh.dispatch_command(do_single_loop)
