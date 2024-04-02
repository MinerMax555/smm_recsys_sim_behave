import os
import matplotlib.pyplot as plt
from helper_files.data_loader import load_data


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

    plt.plot(iteration_range, proportions_dict['us_proportion'], label='US Proportion', color='orange', linestyle='-')

    # Filling the areas under the curves
    plt.fill_between(iteration_range, proportions_dict['us_proportion'], alpha=0.1, color='orange')

    # Plotting the baseline proportions as horizontal lines
    plt.hlines(y=baselines['global_baseline_us'], xmin=iteration_range[0], xmax=iteration_range[-1], colors='orange', linestyles='--', label='Global Baseline US')

    plt.ylim(0, 1)
    plt.xlim(iteration_range[0], iteration_range[-1])
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

    if choice_model_name == 'rank_based':
        choice_model_name = 'Rank Based'

    # Adding labels and title
    plt.title(f'Country Recommendation Distribution (ItemKNN, {choice_model_name})')
    plt.xlabel('Iteration')
    plt.ylabel('Proportion of tracks to US')
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(save_folder, 'proportions_plot.png'))


def plot_jsd(save_folder, iteration_range, jsd_values):
    """
    Plot the progression of average JSD between history and recommendations at each iteration.

    Parameters:
    - iteration_range: A list of iteration numbers.
    - jsd_values: A list of JSD values for each iteration.
    - save_folder: Directory where the plot will be saved.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(iteration_range, jsd_values, label='Average JSD', color='green', linestyle='-')

    plt.title('Progression of JSD between History and Recommendations')
    plt.xlabel('Iteration')
    plt.ylabel('Average JSD')
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)
    plt.legend(loc='upper left')
    plt.xlim(iteration_range[0], iteration_range[-1])
    plt.ylim(0)

    plt.savefig(os.path.join(save_folder, 'jsd_plot.png'))


def plot_main():
    experiments_folder = 'experiments'
    experiment_name = 'example'
    focus_country = 'US'

    plot_save_folder = os.path.join(experiments_folder, experiment_name, 'plots')

    if not plot_save_folder:
        os.makedirs(plot_save_folder)

    # Load the data
    proportions, iterations, baselines, choice_model_name, jsd_values = load_data(experiments_folder, experiment_name, focus_country)

    # Plot the Proportions Plot
    plot_proportions(plot_save_folder, proportions, list(range(1, iterations + 1)), baselines, choice_model_name)
    # Plot the JSD Plot
    plot_jsd(plot_save_folder, list(range(1, iterations + 1)), jsd_values)


if __name__ == "__main__":
    plot_main()
