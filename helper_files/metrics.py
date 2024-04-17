from collections import Counter
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
    global_baseline_focus_country = \
        global_interactions_with_tracks[global_interactions_with_tracks['country'] == focus_country].shape[0] / \
        global_interactions_with_tracks.shape[0]

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

    # Calculate the US proportion per country
    countries = demographics['country'].unique()
    countries.sort()
    for country in countries:
        # Filter for users from the specific country
        country_users = demographics[demographics['country'] == country].index
        country_top_k_data = top_k_data[top_k_data['user_id'].isin(country_users)]

        # Merge with US tracks
        country_us_interactions = country_top_k_data.merge(global_us_tracks, on='item_id', how='inner')

        # Calculate proportion
        country_proportion = len(country_us_interactions) / len(country_top_k_data) if len(
            country_top_k_data) > 0 else 0
        proportion_results.append({
            "model": model,
            "choice_model": choice_model,
            "iteration": iteration,
            "country": country,
            "us_proportion": country_proportion
        })

    proportion_results.append({
        "model": model,
        "choice_model": choice_model,
        "iteration": iteration,
        "country": "global",
        "us_proportion": global_proportion
    })

    return pd.DataFrame(proportion_results)


def join_interaction_with_country(interaction_history, demographics, tracks_info, tracks_with_popularity):
    """
    Join interaction history with demographics to associate each interaction with a country.

    Parameters:
    - interaction_history: DataFrame of interaction histories without country information.
    - demographics: DataFrame of user demographics, including country.
    - tracks_info: DataFrame of tracks information, which is used to ensure item_id compatibility.

    Returns:
    DataFrame of interaction histories enriched with country information.
    """
    # Merge to get country and other demographic details for each interaction
    interaction_history = interaction_history.merge(demographics, left_on='user_id', right_index=True, how='left')
    interaction_history = interaction_history.rename(columns={'country': 'user_country'})

    # Merge to include track details
    interaction_history = interaction_history.merge(tracks_info, on='item_id', how='left')
    interaction_history = interaction_history.rename(columns={'country': 'artist_country'})

    # Merge to include popularity bins
    interaction_history = interaction_history.merge(tracks_with_popularity[['item_id', 'popularity_bin']], on='item_id', how='left')

    return interaction_history


def prepare_jsd_distributions(recs_merged, interactions_merged, all_item_countries):
    """
    Prepare the distributions for JSD calculation.

    Parameters:
    - recs_merged: DataFrame containing top K interactions.
    - interactions_merged: DataFrame containing global interaction history.
    - all_item_countries: List of all countries in the dataset.

    Returns:
    Two numpy arrays representing the distributions of tracks over countries for global interactions and top K interactions.
    """

    # Calculate the distribution for top K interactions (recommendations) based on countries
    top_k_distribution = calculate_country_distribution(recs_merged, all_item_countries)

    return top_k_distribution


def calculate_iteration_jsd_per_user(recs_merged, tracks_info, history_distribution, model, choice_model, iteration, user_ids):
    """
    Calculate the Jensen-Shannon Divergence (JSD) between history and recommendations for each user.

    Parameters:
    - recs_merged: DataFrame containing top K interactions.
    - tracks_info: DataFrame containing tracks information.
    - interactions_merged: DataFrame containing global interaction history.
    - model: The name of the model used in the experiment.
    - choice_model: The name of the choice model used in the experiment.
    - iteration: The iteration number.

    Returns:
    DataFrame with the JSD values per user.
    """
    unique_item_countries = tracks_info['country'].unique()
    jsd_rows_per_user = []

    recs_by_user = recs_merged.groupby('user_id')

    for user_id in range(0, len(user_ids)):
        user_recs = recs_by_user.get_group(user_id)

        top_k_distribution = calculate_country_distribution(user_recs, unique_item_countries)

        jsd_rows_per_user.append({
            'user_id': user_id,
            'user_country': user_recs['user_country'].values[0],
            'jsd': jensenshannon(history_distribution[user_id], top_k_distribution, base=2)
        })

    # Obtain the mean JSD per country
    jsd_raw_df = pd.DataFrame(jsd_rows_per_user)
    # Group by country and aggregate the mean JSD
    jsd_country_df = jsd_raw_df.groupby('user_country').agg(
        jsd=pd.NamedAgg(column='jsd', aggfunc='mean'),
        user_count=pd.NamedAgg(column='user_id', aggfunc='size'),
    ).reset_index()

    # add global row
    global_jsd = jsd_raw_df['jsd'].mean()
    global_row = pd.Series(["global", len(user_ids), global_jsd], index=["user_country", "user_count", "jsd"])
    jsd_country_df = pd.concat([jsd_country_df, global_row.to_frame().T], ignore_index=True)

    jsd_country_df.rename(columns={
        'user_country': 'country'
    }, inplace=True)
    jsd_country_df['model'] = model
    jsd_country_df['choice_model'] = choice_model
    jsd_country_df['iteration'] = iteration

    return jsd_country_df


def calculate_country_distribution(df, country_list):
    """
    Calculates the distribution of tracks over countries.

    Parameters:
    - df: DataFrame with 'item_id'.
    - country_list: List of countries to ensure those not in the dataframe return 0.

    Returns:
    Numpy array representing the distribution of tracks across countries.
    """
    # Ensure that 'tracks_info' only contains the necessary columns to avoid key collisions on merge
    # tracks_info_simplified = tracks_info[['item_id', 'country']]

    # Merge to get country information for each item
    # merged_df = df.merge(tracks_info_simplified, on='item_id', how='left')
    # country_column = 'country_y' if 'country_y' in merged_df.columns else 'country'

    # slightly slower method
    """
    # Calculate proportion of tracks per country using the correct country column
    country_counts = df['artist_country'].value_counts(normalize=True)

    # Ensure distribution includes all countries present in tracks_info, filling missing values with 0
    distribution = country_counts.reindex(country_list, fill_value=0).values
    """

    # faster method, identical to code used before
    country_counter = Counter(df['artist_country'])
    country_counts = pd.Series(country_counter)
    distribution = country_counts.reindex(country_list, fill_value=0).values
    distribution = distribution / distribution.sum()
    return distribution


def create_popularity_bins(interactions_merged, tracks_info):
    """
    Create popularity bins for tracks based on interaction counts.

    Parameters:
    - interactions_merged: DataFrame containing global interaction history.
    - tracks_info: DataFrame containing tracks information.

    Returns:
    DataFrame with popularity bins for each track.
    """
    # Calculate popularity of each track
    popularity_counts = interactions_merged['item_id'].value_counts().rename('interaction_count')
    tracks_with_popularity = tracks_info.merge(popularity_counts, left_on='item_id', right_index=True, how='left')
    tracks_with_popularity['interaction_count'] = tracks_with_popularity['interaction_count'].fillna(0)

    # Calculate quantiles for bin thresholds
    quantiles = tracks_with_popularity['interaction_count'].quantile([0.2, 0.8])

    # Assign bins
    tracks_with_popularity['popularity_bin'] = np.select(
        [
            tracks_with_popularity['interaction_count'] <= quantiles[0.2],
            tracks_with_popularity['interaction_count'] > quantiles[0.8]
        ],
        ['Low', 'High'], default='Medium'
    )

    return tracks_with_popularity


def calculate_user_bin_jsd(recs_merged, interactions_merged):
    """
    Calculate the Jensen-Shannon Divergence (JSD) between history and recommendations for each user, based on popularity bins.

    Parameters:
    - recs_merged: DataFrame containing top K interactions.
    - interactions_merged: DataFrame containing global interaction history.

    Returns:
    DataFrame with the JSD values per user.
    """
    # Get bin counts per user for recommendations and history
    recs_bin_counts = recs_merged.groupby('user_id')['popularity_bin'].value_counts(normalize=True).unstack(fill_value=0)
    hist_bin_counts = interactions_merged.groupby('user_id')['popularity_bin'].value_counts(normalize=True).unstack(fill_value=0)

    # Ensure all bins are represented
    all_bins = ['High', 'Medium', 'Low']
    recs_bin_counts = recs_bin_counts.reindex(columns=all_bins, fill_value=0)
    hist_bin_counts = hist_bin_counts.reindex(columns=all_bins, fill_value=0)

    # Calculate JSD for each user
    jsd_results = (recs_bin_counts.apply(lambda x: jensenshannon(x, hist_bin_counts.loc[x.name], base=2), axis=1)
                   .reset_index().rename(columns={0: 'jsd'}))

    jsd_results['country'] = recs_merged.groupby('user_id')['user_country'].first()

    return jsd_results


def aggregate_jsd_by_country(bin_jsd_df):
    """
    Aggregate JSD by country, computing mean JSD for each country and globally.

    Parameters:
    - bin_jsd_df: DataFrame containing JSD values per user and country.
    """
    aggregated_jsd = bin_jsd_df.groupby('country')['jsd'].mean().reset_index()
    aggregated_jsd.columns = ['country', 'bin_jsd']
    global_jsd = bin_jsd_df['jsd'].mean()
    global_row = pd.DataFrame([['global', global_jsd]], columns=['country', 'bin_jsd'])
    aggregated_jsd = pd.concat([aggregated_jsd, global_row], ignore_index=True)

    return aggregated_jsd


def merge_jsd_dataframes(jsd_df, bin_jsd_df):
    """
    Merge JSD dataframes to include bin JSD values.

    Parameters:
    - jsd_df: DataFrame containing JSD values per user and country.
    - bin_jsd_df: DataFrame containing JSD values per user and country based on popularity bins.

    Returns:
    DataFrame with JSD values per user and country, including bin JSD values.
    """
    aggregated_bin_jsd = aggregate_jsd_by_country(bin_jsd_df)
    jsd_df = jsd_df.merge(aggregated_bin_jsd, on='country', how='left')
    return jsd_df
