import os
import argh
import matplotlib.pyplot as plt
from helper_files.data_loader import load_data


def plot_proportions(save_folder, proportions_dict, iteration_range, baselines, params_dict, focus_country):
    """
    Plot the proportions of local and focus_country (mostly US) track recommendations.

    Parameters:
    - proportions_dict: A dictionary containing the proportions of local and focus_country (mostly US) tracks.
    - iteration_range: A list of iteration numbers.
    - baselines: A dictionary containing the baseline proportions.
    - params_dict: A dictionary containing the parameters used in the experiment.
    - focus_country: The country code for the focus group.

    Returns:
    A plot showing the proportions of local and focus_country (mostly US) track recommendations.
    """

    plt.figure(figsize=(15, 7))
    plt.plot(iteration_range, proportions_dict, label=f'{focus_country} Proportion', color='orange', linestyle='-')

    # Filling the areas under the curves
    plt.fill_between(iteration_range, proportions_dict, alpha=0.1, color='orange')

    # Plotting the baseline proportions as horizontal lines
    plt.hlines(y=baselines['global_baseline_focus_country'], xmin=iteration_range[0], xmax=iteration_range[-1], colors='orange', linestyles='--', label=f'Global Baseline {focus_country}')

    # Ensure the first tick is displayed on the x-axis
    if iteration_range[0] not in plt.xticks()[0]:
        plt.xticks([iteration_range[0]] + list(plt.xticks()[0]))

    plt.xlim(iteration_range[0], iteration_range[-1])
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

    plt.title(f'Country Recommendation Distribution ({params_dict["model"]}, {params_dict["choice_model"]}, {params_dict["dataset_name"]})')
    plt.xlabel('Iteration')
    plt.ylabel(f'Proportion of tracks to {focus_country}')
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(save_folder, 'proportions_plot.png'))


def plot_jsd(save_folder, iteration_range, jsd_values, params_dict, focus_country):
    """
    Plot the progression of average JSD between history and recommendations at each iteration.

    Parameters:
    - iteration_range: A list of iteration numbers.
    - jsd_values: A list of JSD values for each iteration.
    - save_folder: Directory where the plot will be saved.
    - params_dict: A dictionary containing the parameters used in the experiment.
    - focus_country: The country code for the focus group.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(iteration_range, jsd_values, label='Average JSD', color='green', linestyle='-')

    plt.title(f'JSD between History and Recommendations of {focus_country} ({params_dict["model"]}, {params_dict["choice_model"]}, {params_dict["dataset_name"]})')
    plt.xlabel('Iteration')

    # Ensure the first tick is displayed on the x-axis
    if iteration_range[0] not in plt.xticks()[0]:
        plt.xticks([iteration_range[0]] + list(plt.xticks()[0]))

    # Adding labels and title
    plt.ylabel('Average JSD')
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)
    plt.legend(loc='upper left')
    plt.xlim(iteration_range[0], iteration_range[-1])

    plt.savefig(os.path.join(save_folder, 'jsd_plot.png'))


@argh.arg('-ef', '--experiments-folder', type=str, help='Path to the experiments folder')
@argh.arg('-ex', '--experiment-name', type=str, help='Name of the specific experiment')
@argh.arg('-fc', '--focus-country', type=str, help='Focus country code')
def plot_main(experiments_folder="experiments", experiment_name="sample1", focus_country="US"):
    plot_save_folder = os.path.join(experiments_folder, experiment_name, 'plots')

    if not os.path.exists(plot_save_folder):
        os.makedirs(plot_save_folder)

    print("Loading data...")

    # Load the data
    proportions, iterations, baselines, params_dict, jsd_values = load_data(experiments_folder, experiment_name, focus_country)

    if params_dict["choice_model"] == 'rank_based':
        params_dict["choice_model"] = 'Rank Based'
    elif params_dict["choice_model"] == 'us_centric':
        params_dict["choice_model"] = 'US Centric'

    # This intentionally excludes the last iteration beacuse for that there is only the dataset and no recommendations
    iterations = list(range(1, iterations))

    # Plot the Proportions Plot
    plot_proportions(plot_save_folder, proportions, iterations, baselines, params_dict, focus_country)
    # Plot the JSD Plot
    plot_jsd(plot_save_folder, iterations, jsd_values, params_dict, focus_country)


if __name__ == "__main__":
    argh.dispatch_command(plot_main)
