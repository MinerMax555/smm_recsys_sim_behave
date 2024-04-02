import glob
import os
import pandas as pd
from plot_helper import calculate_global_and_country_specific_baselines, calculate_proportions

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
    A tuple containing the proportions dictionary, the number of iterations, and the baseline proportions.
    """

    proportions = {
        'us_proportion': [],
        'global_baseline_us': [],
        'country_specific_baseline_us': []
    }

    choice_model_name = None

    input_dir_path = os.path.join(experiments_folder, experiment_name, 'input')

    log_dir = os.path.join(os.path.join(experiments_folder, experiment_name), 'log', 'iteration_1', 'ItemKNN')

    log_files = glob.glob(os.path.join(log_dir, 'ItemKNN-dataset-*.log'))

    if log_files:
        with open(log_files[0], 'r') as f:
            for line in f:
                if '--choice-model' in line:
                    pass
                    choice_model_name = line.split('--choice-model')[1].split("'")[2]
                    break

    dataset_inter_filepath = os.path.join(input_dir_path, 'dataset.inter')
    demographics_filepath = os.path.join(input_dir_path, 'demographics.tsv')
    tracks_filepath = os.path.join(input_dir_path, 'tracks.tsv')

    global_interactions = pd.read_csv(dataset_inter_filepath, delimiter='\t', header=None, skiprows=1, names=['user_id', 'item_id'])
    demographics_info = pd.read_csv(demographics_filepath, delimiter='\t', header=None, names=['country', 'user_id', 'gender', 'timestamp'])

    # Load tracks.tsv, reset the index to use as item_id, and assign column names accordingly
    tracks_info = pd.read_csv(tracks_filepath, delimiter='\t', header=None)
    tracks_info.reset_index(inplace=True)
    tracks_info.columns = ['item_id', 'artist', 'title', 'country']

    global_interactions['item_id'] = global_interactions['item_id'].astype(int)
    tracks_info['item_id'] = tracks_info['item_id'].astype(int)

    baselines = calculate_global_and_country_specific_baselines(global_interactions, demographics_info, tracks_info, focus_country)

    # read the number of iterations from the output folder
    iterations = len(os.listdir(os.path.join(experiments_folder, experiment_name, 'datasets'))) - 1

    # The loop to calculate proportions should only work with 'local_proportion' and 'us_proportion'
    for iteration in range(1, iterations + 1):
        interaction_history = load_interaction_history(os.path.join(experiments_folder, experiment_name), iteration)
        iteration_proportions = calculate_proportions(interaction_history, tracks_info, baselines, focus_country)

        proportions['us_proportion'].append(iteration_proportions['us_proportion'])

    return proportions, iterations, baselines, choice_model_name

