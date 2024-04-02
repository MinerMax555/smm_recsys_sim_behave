import numpy as np
import pandas as pd
from tqdm import tqdm



def prefilter_recommendations(
        recommendations: pd.DataFrame,
        demographics: pd.DataFrame,
        tracks: pd.DataFrame,
        control_country: str | None = None
):
    filtered_recs = recommendations.copy()
    if control_country:
        # Remove any recommendations for users in the control country
        frozen_users = list(demographics.loc[demographics['country'] == control_country])
        filtered_recs = filtered_recs[~filtered_recs['user_id'].isin(frozen_users)]

    return filtered_recs


def choice_model_random(recommendations: pd.DataFrame):
    acc_list = []
    for user_id in tqdm(recommendations['user_id'].unique(), desc='Applying choice model'):
        recs = recommendations.loc[recommendations['user_id'] == user_id]
        # Randomly choose a song of these
        acc_list.append([user_id, recs.sample(1)['item_id'].values[0]])

    return pd.DataFrame(acc_list, columns=['user_id', 'item_id'], dtype=int)


def choice_model_rank_based(recommendations: pd.DataFrame, alpha: float = 0.1):
    acc_list = []
    for user_id in tqdm(recommendations['user_id'].unique(), desc='Applying choice model'):
        recs = recommendations.loc[recommendations['user_id'] == user_id]
        p_list = [np.exp(-alpha * i) for i in range(1, len(recs) + 1)]
        # normalize probabilities
        p_list = np.array(p_list) / np.sum(p_list)
        chosen_rec = np.random.choice(range(0, len(recs)), p=p_list)
        acc_list.append([user_id, recs.iloc[chosen_rec]['item_id']])

    return pd.DataFrame(acc_list, columns=['user_id', 'item_id'], dtype=int)


def country_centric(recommendations: pd.DataFrame, tracks: pd.DataFrame, country='US', non_country_chance=0.0,
                    invert=False):
    acc_list = []
    for user_id in tqdm(recommendations['user_id'].unique(), desc='Applying choice model'):
        recs = recommendations.loc[recommendations['user_id'] == user_id]
        # Item_id in recs
        from_country = tracks.iloc[recs['item_id']]['country'] == country
        # Chance of 1 for songs from the country, chance of non_country_chance for songs not from the country

        if invert:
            # Inverted mode: instead of focusing on country songs, focus on non-country songs
            p_list = [non_country_chance if x else 1 for x in from_country]
        else:
            p_list = [1 if x else non_country_chance for x in from_country]

        if np.max(p_list) == 0:
            # no suitable song in the recommendations -> don't accept any for this user
            continue

        # normalize probabilities
        p_list = np.array(p_list) / np.sum(p_list)
        # Sample by probability defined before
        choice = np.random.choice(range(0, len(recs)), p=p_list)
        acc_list.append([user_id, recs.iloc[choice]['item_id']])

    return pd.DataFrame(acc_list, columns=['user_id', 'item_id'], dtype=int)


def accept_new_recommendations(choice_model: str,
                               recommendations: pd.DataFrame,
                               demographics: pd.DataFrame,
                               tracks: pd.DataFrame,
                               k: int = 10):
    if choice_model == 'random':
        recommendations = choice_model_random(recommendations)
    elif choice_model == 'rank_based':
        recommendations = choice_model_rank_based(recommendations)
    elif choice_model == 'us_centric':
        recommendations = country_centric(recommendations, tracks, country='US', non_country_chance=0.0)
    elif choice_model == 'non_us_centric':
        recommendations = country_centric(recommendations, tracks, country='US', non_country_chance=0.0, invert=True)
    else:
        raise NotImplementedError('Unknown Choice model!')

    return recommendations
