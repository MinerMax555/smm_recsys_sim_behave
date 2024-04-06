import glob
import json
import os
import pandas as pd
from tqdm import tqdm

from helper_files.metrics import calculate_global_baseline, calculate_proportions, calculate_iteration_jsd


def load_interaction_history(experiment_dir, iteration_number):
    """
    Loads the interaction history for a given iteration.

    Parameters:
    - experiment_dir: The directory where the experiment data is stored.
    - iteration_number: The iteration number to load the interaction history for.

    Returns:
    A pandas DataFrame containing the interaction history.
    """
    # Construct the file path for the interaction history file
    interaction_file = f'iteration_{iteration_number}.inter'
    interaction_path = os.path.join(experiment_dir, 'datasets', interaction_file)

    interaction_history = pd.read_csv(interaction_path, delimiter='\t', header=None, skiprows=1, names=['user_id', 'item_id'])

    return interaction_history


def load_data(experiments_folder, experiment_name, focus_country):
    """
    Load the data for the specified experiment.

    Parameters:
    - experiments_folder: The folder containing the experiment data.
    - experiment_name: The name of the experiment to load.
    - focus_country: The country code for the focus group.

    Returns:
    A tuple containing the proportions dictionary, the number of iterations, the baseline proportions and the jsd values.
    """

    proportions = {
        'us_proportion': [],
        'global_baseline_focus_country': [],
    }

    params_dict = {}

    jsd_values = []

    input_dir_path = os.path.join(experiments_folder, experiment_name, 'input')

    # get choice model name by reading the params.json
    with open(os.path.join(experiments_folder, experiment_name, 'params.json')) as f:
        params = json.load(f)
        params_dict["model"] = params["model"]
        params_dict["choice_model"] = params["choice_model"]
        params_dict["dataset_name"] = params["dataset_name"]


    dataset_inter_filepath = os.path.join(input_dir_path, 'dataset.inter')
    tracks_filepath = os.path.join(input_dir_path, 'tracks.tsv')

    global_interactions = pd.read_csv(dataset_inter_filepath, delimiter='\t', header=None, skiprows=1, names=['user_id', 'item_id'])

    # Load tracks.tsv, reset the index to use as item_id, and assign column names accordingly
    tracks_info = pd.read_csv(tracks_filepath, delimiter='\t', header=None)
    tracks_info.reset_index(inplace=True)
    tracks_info.columns = ['item_id', 'artist', 'title', 'country']

    global_interactions['item_id'] = global_interactions['item_id'].astype(int)
    tracks_info['item_id'] = tracks_info['item_id'].astype(int)

    baselines = calculate_global_baseline(global_interactions, tracks_info, focus_country)

    # read the number of iterations from the output folder
    iterations = len(os.listdir(os.path.join(experiments_folder, experiment_name, 'datasets'))) - 1

    # The loop to calculate proportions
    for iteration in tqdm(range(1, iterations + 1), desc='Calculating proportions per iteration'):
        interaction_history = load_interaction_history(os.path.join(experiments_folder, experiment_name), iteration)
        iteration_proportions = calculate_proportions(interaction_history, tracks_info, baselines, focus_country)

        proportions['us_proportion'].append(iteration_proportions['us_proportion'])

        jsd_values.append(calculate_iteration_jsd(interaction_history, global_interactions, tracks_info))

    return proportions, iterations, baselines, params_dict, jsd_values

