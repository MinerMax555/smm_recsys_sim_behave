from time import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm


def calculate_global_baseline(global_interactions, tracks_info, focus_country):
    """
    Calculate the global and country-specific baseline proportions.

    Parameters:
    - interactions_merged: A pandas DataFrame containing global interaction history.
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


def calculate_us_proportion(interaction_data, tracks_info):
    """
    Helper function to calculate the proportion of US tracks in the interaction data.

    Parameters:
    - interaction_data: DataFrame containing interactions data.
    - tracks_info: DataFrame containing tracks and their country of origin.

    Returns:
    The proportion of US tracks in the interaction data.
    """
    us_tracks = tracks_info[tracks_info['country'] == 'US']
    merged_data = interaction_data.merge(us_tracks, on='item_id', how='inner')
    us_proportion = len(merged_data) / len(interaction_data) if len(interaction_data) > 0 else 0
    return us_proportion


def calculate_proportions(top_k_data, tracks_info, demographics, model, choice_model, iteration):
    """
    Calculate the proportions of US track recommendations for each country and globally.

    Parameters:
    - recs_merged: DataFrame containing the top K interactions for a specific iteration.
    - tracks_info: DataFrame containing tracks and their country of origin.
    - demographics: DataFrame containing user demographics, indexed by user_id.
    - model: The name of the model used in the experiment.
    - choice_model: The name of the choice model used in the experiment.
    - iteration: The iteration number.

    Returns:
    List of dictionaries with US proportion values for each country and globally.
    """
    # This holds the combined results
    proportion_results = []

    # Calculate the proportion for each country and globally
    global_us_tracks = tracks_info[tracks_info['country'] == 'US']
    global_us_interactions = top_k_data.merge(global_us_tracks, on='item_id', how='inner')

    global_proportion = len(global_us_interactions) / len(top_k_data) if len(top_k_data) > 0 else 0
    proportion_results.append({
        "model": model,
        "choice_model": choice_model,
        "iteration": iteration,
        "country": "global",
        "us_proportion": global_proportion
    })

    # Calculate the US proportion per country
    for country in demographics['country'].unique():
        # Filter for users from the specific country
        country_users = demographics[demographics['country'] == country].index
        country_top_k_data = top_k_data[top_k_data['user_id'].isin(country_users)]

        # Merge with US tracks
        country_us_interactions = country_top_k_data.merge(global_us_tracks, on='item_id', how='inner')

        # Calculate proportion
        country_proportion = len(country_us_interactions) / len(country_top_k_data) if len(country_top_k_data) > 0 else 0
        proportion_results.append({
            "model": model,
            "choice_model": choice_model,
            "iteration": iteration,
            "country": country,
            "us_proportion": country_proportion
        })

    return proportion_results


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
    merged_df = interaction_history.merge(demographics, left_on='user_id', right_index=True, how='left')
    merged_df = merged_df.rename(columns={'country': 'user_country'})

    merged_df = merged_df.merge(tracks_info, on='item_id', how='left')
    merged_df = merged_df.rename(columns={'country': 'artist_country'})
    return merged_df


def prepare_jsd_distributions(recs_merged, interactions_merged, tracks_info):
    """
    Prepares distributions for JSD calculation, for a specific country or globally, based on track country distribution.

    Parameters are the same as described previously.
    """
    # Calculate the distribution for global interactions (history) based on countries
    global_distribution = calculate_country_distribution(interactions_merged, tracks_info)

    # Calculate the distribution for top K interactions (recommendations) based on countries
    top_k_distribution = calculate_country_distribution(recs_merged, tracks_info)

    return global_distribution, top_k_distribution



def calculate_iteration_jsd(recs_merged, tracks_info, interactions_merged, model, choice_model, iteration):
    """
    Calculate the JSD for each country and globally for a given iteration, using interaction history.

    Parameters:
    - recs_merged: DataFrame containing the recommendations merged with user demographics and track information.
    - tracks_info: DataFrame with tracks information.
    - interactions_merged: DataFrame containing the original interaction history merged with user demographics and track information.
    - model: The name of the model used in the experiment.
    - choice_model: The name of the choice model used in the experiment.
    - iteration: The iteration number.

    Returns:
    Dictionary with JSD values for each country and a global JSD value.
    """
    unique_user_countries = recs_merged['user_country'].unique()
    jsd_rows = []

    # Calculate global JSD
    global_history_distribution, global_recommendations_distribution = prepare_jsd_distributions(recs_merged, interactions_merged, tracks_info)
    jsd_rows.append({
        "model": model,
        "choice_model": choice_model,
        "iteration": iteration,
        "country": "global",
        "jsd": jensenshannon(global_history_distribution, global_recommendations_distribution, base=2)
    })

    # Calculate JSD for each country
    for country in unique_user_countries:
        # Filter interactions for the current country
        country_recs = recs_merged.query(f'user_country == "{country}"')
        country_interactions = interactions_merged.query(f'user_country == "{country}"')

        # Prepare distributions for JSD calculation
        history_distribution, recommendations_distribution = prepare_jsd_distributions(country_recs, country_interactions, tracks_info)

        jsd_rows.append({
            "model": model,
            "choice_model": choice_model,
            "iteration": iteration,
            "country": country,
            "jsd": jensenshannon(history_distribution, recommendations_distribution, base=2)
        })

    return jsd_rows


def calculate_country_distribution(df, tracks_info):
    """
    Calculates the distribution of tracks over countries.

    Parameters:
    - df: DataFrame with 'item_id'.
    - tracks_info: DataFrame with track information including 'country'.

    Returns:
    Numpy array representing the distribution of tracks across countries.
    """
    # Ensure that 'tracks_info' only contains the necessary columns to avoid key collisions on merge
    # tracks_info_simplified = tracks_info[['item_id', 'country']]

    # Merge to get country information for each item
    #merged_df = df.merge(tracks_info_simplified, on='item_id', how='left')
    #country_column = 'country_y' if 'country_y' in merged_df.columns else 'country'

    # Calculate proportion of tracks per country using the correct country column
    country_counts = df['artist_country'].value_counts(normalize=True)

    # Ensure distribution includes all countries present in tracks_info, filling missing values with 0
    all_countries = tracks_info['country'].unique()
    distribution = country_counts.reindex(all_countries, fill_value=0).values

    return distribution


