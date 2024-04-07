import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def calculate_global_baseline(global_interactions, tracks_info, focus_country):
    """
    Calculate the global and country-specific baseline proportions.

    Parameters:
    - global_interactions: A pandas DataFrame containing global interaction history.
    - tracks_info: A pandas DataFrame containing tracks and their country of origin, with a new 'item_id' column.
    - focus_country: The country code for the focus group.

    Returns:
    A dictionary containing the global and country-specific baseline proportions.
    """

    # Merge global interactions with track information
    global_interactions_with_tracks = global_interactions.merge(tracks_info, on='item_id')

    # Calculate global baseline proportions
    global_baseline_focus_country = global_interactions_with_tracks[global_interactions_with_tracks['country'] == focus_country].shape[0] / global_interactions_with_tracks.shape[0]

    return {'global_baseline_focus_country': global_baseline_focus_country}


def calculate_proportions(interaction_history, tracks_info, baselines, focus_country):
    """
    Calculate the proportions of focus_country (mostly US) track recommendations.

    Parameters:
    - interaction_history: A pandas DataFrame containing the interaction history.
    - tracks_info: A pandas DataFrame containing tracks and their country of origin.
    - baselines: A dictionary containing the calculated baseline proportions.
    - focus_country: The country code for the focus group.

    Returns:
    A dictionary containing the calculated proportions.
    """

    interaction_history['item_id'] = pd.to_numeric(interaction_history['item_id'], errors='coerce')

    # Join the interaction history with the track country information
    interaction_with_country = interaction_history.merge(tracks_info, left_on='item_id', right_on='item_id', how='left')

    # Calculate the total number of recommendations
    total_recommendations = len(interaction_with_country)

    # Calculate the proportion of focus_country (mostly US) tracks
    us_count = interaction_with_country[interaction_with_country['country'] == focus_country].shape[0]
    us_proportion = us_count / total_recommendations if total_recommendations else 0

    # Include the baseline proportions
    proportions = {'us_proportion': us_proportion}

    proportions.update(baselines)

    return proportions

def join_interaction_with_country(interaction_history, demographics, tracks_info):
    """
    Join interaction history with demographics to associate each interaction with a country.

    Parameters:
    - interaction_history: DataFrame of interaction histories without country information.
    - demographics: DataFrame of user demographics, including country.
    - tracks_info: DataFrame of tracks information, which is used to ensure item_id compatibility.

    Returns:
    DataFrame of interaction histories enriched with country information.
    """
    # Ensure correct data types for merging
    interaction_history['user_id'] = interaction_history['user_id'].astype(int)
    demographics.index = demographics.index.astype(int)
    tracks_info['item_id'] = tracks_info['item_id'].astype(int)

    # Merge to get country for each interaction
    interaction_with_country = interaction_history.merge(demographics, left_on='user_id', right_index=True, how='left')

    return interaction_with_country


def prepare_jsd_distributions(top_k_data, global_interactions, tracks_info, country=None):
    """
    Prepares distributions for JSD calculation, for a specific country or globally.

    Parameters:
    - top_k_data: DataFrame containing the top K interactions for a specific iteration.
    - global_interactions: DataFrame containing the global interactions.
    - tracks_info: DataFrame containing tracks and their country of origin.
    - country: The specific country code to filter by, or None for global distributions.

    Returns:
    Two arrays representing the distributions of history and recommendations for JSD calculation.
    """
    unique_items = tracks_info['item_id'].unique()

    # Calculate the distribution for global interactions (history)
    global_distribution = calculate_distribution(global_interactions, unique_items)

    # If a country filter is applied, calculate the distribution for the country-specific interactions
    if country:
        top_k_filtered = top_k_data[top_k_data['country'] == country]
    else:
        top_k_filtered = top_k_data

    # Calculate the distribution for top K interactions (recommendations)
    top_k_distribution = calculate_distribution(top_k_filtered, unique_items)

    return global_distribution, top_k_distribution


def calculate_iteration_jsd(interaction_with_country, tracks_info, global_interactions, model, choice_model, iteration):
    """
    Calculate the JSD for each country and globally for a given iteration, using interaction history.

    Parameters:
    - interaction_with_country: DataFrame containing the interaction history with associated country information.
    - tracks_info: DataFrame with tracks information.
    - global_interactions: DataFrame containing global interactions.
    - model: The name of the model used in the experiment.
    - choice_model: The name of the choice model used in the experiment.
    - iteration: The iteration number.

    Returns:
    Dictionary with JSD values for each country and a global JSD value.
    """
    unique_countries = interaction_with_country['country'].unique()
    jsd_rows = []

    # Calculate global JSD
    global_history_distribution, global_recommendations_distribution = prepare_jsd_distributions(interaction_with_country, global_interactions, tracks_info)
    jsd_rows.append({
        "model": model,
        "choice_model": choice_model,
        "iteration": iteration,
        "country": "global",
        "jsd": jensenshannon(global_history_distribution, global_recommendations_distribution, base=2)
    })

    # Calculate JSD for each country
    for country in unique_countries:
        # Filter interactions for the current country
        country_interaction_history = interaction_with_country[interaction_with_country['country'] == country]

        # Prepare distributions for JSD calculation
        history_distribution, recommendations_distribution = prepare_jsd_distributions(country_interaction_history, global_interactions, tracks_info, country=country)

        jsd_rows.append({
            "model": model,
            "choice_model": choice_model,
            "iteration": iteration,
            "country": country,
            "jsd": jensenshannon(history_distribution, recommendations_distribution, base=2)
        })

    return jsd_rows


def calculate_distribution(df, unique_items):
    """
    Calculates the distribution of items over the unique items.

    Parameters:
    - df: DataFrame with 'item_id'.
    - unique_items: Numpy array of unique item IDs.

    Returns:
    Numpy array representing the distribution of items.
    """
    item_counts = df['item_id'].value_counts(normalize=True)
    distribution = item_counts.reindex(unique_items, fill_value=0).values

    return distribution
