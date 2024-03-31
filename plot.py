import glob
import os
import matplotlib.pyplot as plt
import pandas as pd


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

def calculate_global_and_country_specific_baselines(global_interactions, demographics_info, tracks_info, control_country, focus_country):
    """
    Calculate the global and country-specific baseline proportions.

    Parameters:
    - global_interactions: A pandas DataFrame containing global interaction history.
    - demographics_info: A pandas DataFrame containing user demographics.
    - tracks_info: A pandas DataFrame containing tracks and their country of origin, with a new 'item_id' column.

    Returns:
    A dictionary containing the global and country-specific baseline proportions.
    """

    # Merge global interactions with track information
    global_interactions_with_tracks = global_interactions.merge(tracks_info, on='item_id')

    # Calculate global baseline proportions
    global_baseline_local = global_interactions_with_tracks[global_interactions_with_tracks['country'] == control_country].shape[0] / global_interactions_with_tracks.shape[0]
    global_baseline_us = global_interactions_with_tracks[global_interactions_with_tracks['country'] == focus_country].shape[0] / global_interactions_with_tracks.shape[0]

    de_users_in_demographics = demographics_info[demographics_info['country'] == control_country]
    country_specific_interactions = global_interactions[global_interactions['user_id'].isin(de_users_in_demographics['user_id'])]
    country_specific_interactions_with_tracks = country_specific_interactions.merge(tracks_info, on='item_id')

    # Calculate country-specific baseline proportions
    country_specific_baseline_local = country_specific_interactions_with_tracks[country_specific_interactions_with_tracks['country'] == control_country].shape[0] / country_specific_interactions_with_tracks.shape[0]
    country_specific_baseline_us = country_specific_interactions_with_tracks[country_specific_interactions_with_tracks['country'] == focus_country].shape[0] / country_specific_interactions_with_tracks.shape[0]

    print(global_baseline_local, global_baseline_us, country_specific_baseline_local, country_specific_baseline_us)

    return {
        'global_baseline_local': global_baseline_local,
        'global_baseline_us': global_baseline_us,
        'country_specific_baseline_local': country_specific_baseline_local,
        'country_specific_baseline_us': country_specific_baseline_us
    }


def calculate_proportions(interaction_history, tracks_info, baselines, control_country, focus_country):
    """
    Calculate the proportions of local and US track recommendations.

    Parameters:
    - interaction_history: A pandas DataFrame containing the interaction history.
    - tracks_info: A pandas DataFrame containing tracks and their country of origin.
    - baselines: A dictionary containing the calculated baseline proportions.

    Returns:
    A dictionary containing the calculated proportions.
    """

    interaction_history['item_id'] = pd.to_numeric(interaction_history['item_id'], errors='coerce')

    # Join the interaction history with the track country information
    interaction_with_country = interaction_history.merge(tracks_info, left_on='item_id', right_on='item_id', how='left')

    # Calculate the total number of recommendations
    total_recommendations = len(interaction_with_country)

    # Calculate the proportion of local (DE) tracks
    local_count = interaction_with_country[interaction_with_country['country'] == control_country].shape[0]
    local_proportion = local_count / total_recommendations if total_recommendations else 0

    # Calculate the proportion of US tracks
    us_count = interaction_with_country[interaction_with_country['country'] == focus_country].shape[0]
    us_proportion = us_count / total_recommendations if total_recommendations else 0

    # Include the baseline proportions
    proportions = {
        'local_proportion': local_proportion,
        'us_proportion': us_proportion
    }

    proportions.update(baselines)

    return proportions


# Module for plotting
def plot_proportions(save_folder, proportions_dict, iteration_range, baselines, choice_model_name):
    """
    Plot the proportions of local and US track recommendations.

    Parameters:
    - proportions_dict: A dictionary containing the proportions of local and US tracks.
    - iteration_range: A list of iteration numbers.
    - baselines: A dictionary containing the baseline proportions.

    Returns:
    A plot showing the proportions of local and US track recommendations.
    """


    plt.figure(figsize=(15, 7))

    # Inverting local_proportion just for plotting purposes
    inverted_local_proportion = [1 - p for p in proportions_dict['local_proportion']]

    # Plotting the actual inverted local and US proportions
    plt.plot(iteration_range, inverted_local_proportion, label='Local Proportion', color='blue', linestyle='-')
    plt.plot(iteration_range, proportions_dict['us_proportion'], label='US Proportion', color='orange', linestyle='-')

    # Filling the areas under the curves
    plt.fill_between(iteration_range, 1, inverted_local_proportion, alpha=0.1, color='blue')
    plt.fill_between(iteration_range, proportions_dict['us_proportion'], alpha=0.1, color='orange')

    # Plotting the baseline proportions as horizontal lines
    plt.hlines(y=1 - baselines['global_baseline_local'], xmin=iteration_range[0], xmax=iteration_range[-1], colors='blue', linestyles='--', label='Global Baseline Local')
    plt.hlines(y=baselines['global_baseline_us'], xmin=iteration_range[0], xmax=iteration_range[-1], colors='orange', linestyles='--', label='Global Baseline US')
    plt.hlines(y=1 - baselines['country_specific_baseline_local'], xmin=iteration_range[0], xmax=iteration_range[-1], colors='blue', linestyles='-.', label='Country Specific Baseline Local')
    plt.hlines(y=baselines['country_specific_baseline_us'], xmin=iteration_range[0], xmax=iteration_range[-1], colors='orange', linestyles='-.', label='Country Specific Baseline US')

    plt.ylim(0, 1)
    plt.xlim(iteration_range[0], iteration_range[-1])
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

    if choice_model_name == 'rank_based':
        choice_model_name = 'Rank Based'

    # Adding labels and title
    plt.title(f'Country Recommendation Distribution (DE, ItemKNN, {choice_model_name})')
    plt.xlabel('Iteration')
    plt.ylabel('Proportion of tracks to US')
    plt.legend(loc='upper right')

    if not os.path.exists(os.path.join(save_folder, 'plots')):
        os.makedirs(os.path.join(save_folder, 'plots'))

    plt.savefig(os.path.join(save_folder, 'plots/proportions_plot.png'))


def load_data(experiments_folder, experiment_name, control_country, focus_country):
    """
    Load the data for the specified experiment.

    Parameters:
    - experiments_folder: The folder containing the experiment data.
    - experiment_name: The name of the experiment to load.
    - control_country: The country code for the control group.
    - focus_country: The country code for the focus group.

    Returns:
    A tuple containing the proportions dictionary, the number of iterations, and the baseline proportions.
    """

    proportions = {
        'local_proportion': [],
        'us_proportion': [],
        'global_baseline_local': [],
        'global_baseline_us': [],
        'country_specific_baseline_local': [],
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

    baselines = calculate_global_and_country_specific_baselines(global_interactions, demographics_info, tracks_info, control_country, focus_country)

    # read the number of iterations from the output folder
    iterations = len(os.listdir(os.path.join(experiments_folder, experiment_name, 'datasets'))) - 1

    # The loop to calculate proportions should only work with 'local_proportion' and 'us_proportion'
    for iteration in range(1, iterations + 1):
        interaction_history = load_interaction_history(os.path.join(experiments_folder, experiment_name), iteration)
        iteration_proportions = calculate_proportions(interaction_history, tracks_info, baselines, control_country, focus_country)

        proportions['local_proportion'].append(iteration_proportions['local_proportion'])
        proportions['us_proportion'].append(iteration_proportions['us_proportion'])
        # print(f"Iteration {iteration} proportions:", iteration_proportions)

    return proportions, iterations, baselines, choice_model_name


def plot_main():
    experiments_folder = 'experiments'
    experiment_name = 'example'
    control_country = 'DE'
    focus_country = 'US'

    # Load the data
    proportions, iterations, baselines,choice_model_name = load_data(experiments_folder, experiment_name, control_country, focus_country)

    # Plot the Proportions Plot
    plot_proportions(os.path.join(experiments_folder, experiment_name), proportions, list(range(1, iterations + 1)), baselines, choice_model_name)


if __name__ == "__main__":
    plot_main()
