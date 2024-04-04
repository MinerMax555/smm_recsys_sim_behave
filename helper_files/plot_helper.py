import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def calculate_global_and_country_specific_baselines(global_interactions, demographics_info, tracks_info, focus_country):
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
    global_baseline_us = global_interactions_with_tracks[global_interactions_with_tracks['country'] == focus_country].shape[0] / global_interactions_with_tracks.shape[0]

    return {'global_baseline_us': global_baseline_us}


def calculate_proportions(interaction_history, tracks_info, baselines, focus_country):
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
    local_count = interaction_with_country[interaction_with_country['country'] != focus_country].shape[0]
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


def calculate_iteration_jsd(interaction_history, recommendations, tracks_info):
    history_with_country = interaction_history.merge(tracks_info, on='item_id', how='left')
    recommendations_with_country = recommendations.merge(tracks_info, on='item_id', how='left')

    unique_countries = tracks_info['country'].unique()

    history_country_counts = history_with_country['country'].value_counts(normalize=True)
    history_distribution = history_country_counts.reindex(unique_countries, fill_value=0).values

    recommendations_country_counts = recommendations_with_country['country'].value_counts(normalize=True)
    recommendations_distribution = recommendations_country_counts.reindex(unique_countries, fill_value=0).values

    # Calculate JSD
    jsd = jensenshannon(history_distribution, recommendations_distribution, base=2)

    return jsd