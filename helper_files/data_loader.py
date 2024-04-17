import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from helper_files.metrics import calculate_global_baseline, calculate_proportions, join_interaction_with_country, \
    calculate_iteration_jsd_per_user, calculate_user_bin_jsd, create_popularity_bins, merge_jsd_dataframes, calculate_country_distribution


def load_top_k_data(experiment_dir, iteration_number):
    """
    Load the top-k recommendations for a given iteration.

    Parameters:
    - experiment_dir: The directory containing the experiment data.
    - iteration_number: The iteration number to load.

    Returns:
    DataFrame containing the top-k recommendations for the given iteration.
    """
    top_k_file = f'output/iteration_{iteration_number}_top_k.tsv'
    top_k_path = os.path.join(experiment_dir, top_k_file)

    top_k_data = pd.read_csv(top_k_path, delimiter='\t', usecols=['user_id', 'item_id'])
    return top_k_data


def load_data(experiments_folder, experiment_name, focus_country):
    input_dir_path = os.path.join(experiments_folder, experiment_name, 'input')
    params_path = os.path.join(experiments_folder, experiment_name, 'params.json')
    demographics_file = os.path.join(input_dir_path, 'demographics.tsv')
    dataset_inter_filepath = os.path.join(input_dir_path, 'dataset.inter')
    tracks_filepath = os.path.join(input_dir_path, 'tracks.tsv')

    with open(params_path) as f:
        params_dict = json.load(f)

    # Load global interactions and tracks info
    global_interactions = pd.read_csv(dataset_inter_filepath, delimiter='\t', header=None, skiprows=1, names=['user_id', 'item_id'])
    tracks_info = pd.read_csv(tracks_filepath, delimiter='\t', header=None).reset_index()
    tracks_info.columns = ['item_id', 'artist', 'title', 'country']
    demographics = pd.read_csv(demographics_file, delimiter='\t', header=None, names=['country', 'age', 'gender', 'signup_date'])

    tracks_with_popularity = create_popularity_bins(global_interactions, tracks_info)

    baselines = calculate_global_baseline(global_interactions, tracks_info, focus_country)
    original_interactions_merged = join_interaction_with_country(global_interactions, demographics, tracks_info, tracks_with_popularity)

    # Calculate the number of iterations
    iterations = len(os.listdir(os.path.join(experiments_folder, experiment_name, 'datasets')))
    iterations = 3

    jsd_data = calculate_prop_jsd(experiments_folder, experiment_name, iterations, tracks_info, demographics, params_dict, original_interactions_merged, tracks_with_popularity)

    global_jsd_df = jsd_data[jsd_data['country'] == focus_country]['jsd'].tolist()
    proportion_df = jsd_data[jsd_data['country'] == 'global']['us_proportion'].tolist()  # We could pick any country here, as they are all the same

    return proportion_df, iterations, baselines, params_dict, global_jsd_df


def calculate_prop_jsd(experiments_folder, experiment_name, iterations, tracks_info, demographics, params_dict, original_interactions_merged, tracks_with_popularity):
    column_dtypes = {
        'model': 'str',
        'choice_model': 'str',
        'iteration': 'int',
        'country': 'str',
        'user_count': 'int',
        'jsd': 'float',
        'us_proportion': 'float'
    }
    jsd_data = pd.DataFrame(columns=column_dtypes.keys()).astype(column_dtypes)

    unique_item_countries = tracks_info['country'].unique()
    user_ids = original_interactions_merged['user_id'].unique()
    interactions_by_user = original_interactions_merged.groupby('user_id')
    history_distribution = np.zeros((len(user_ids), len(unique_item_countries)))

    for user_id in range(0, len(user_ids)):
        user_interactions = interactions_by_user.get_group(user_id)
        history_distribution[user_id] = calculate_country_distribution(user_interactions, unique_item_countries)

    # If csv does exist, load it, else calculate it and save the data.
    if os.path.exists(os.path.join(experiments_folder, experiment_name, 'metrics.csv')):
        print('Loading JSD values from CSV')
        jsd_data = pd.read_csv(os.path.join(experiments_folder, experiment_name, 'metrics.csv'))
    else:
        print('Calculating JSD values and recommendation proportions. This may take a while...')
        for iteration in tqdm(range(1, iterations), desc='Processing Iterations'):
            top_k_data = load_top_k_data(os.path.join(experiments_folder, experiment_name), iteration)
            proportion_df = calculate_proportions(top_k_data, tracks_info, demographics, params_dict["model"], params_dict["choice_model"], iteration)
            recs_merged = join_interaction_with_country(top_k_data, demographics, tracks_info, tracks_with_popularity)

            jsd_df = calculate_iteration_jsd_per_user(recs_merged, tracks_info, history_distribution, params_dict["model"], params_dict["choice_model"], iteration, user_ids)

            bin_jsd_df = calculate_user_bin_jsd(recs_merged, original_interactions_merged)

            jsd_df = merge_jsd_dataframes(jsd_df, bin_jsd_df)
            jsd_df['us_proportion'] = proportion_df['us_proportion']
            jsd_df['local_proportion'] = proportion_df['local_proportion']

            jsd_data = pd.concat([jsd_data, jsd_df], ignore_index=True)

        # Save JSD values to CSV
        csv_save_path = os.path.join(experiments_folder, experiment_name, 'metrics.csv')
        jsd_data.to_csv(csv_save_path, index=False)

        print("Loaded data successfully")

    return jsd_data
