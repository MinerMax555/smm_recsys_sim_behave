import pandas as pd
from tqdm import tqdm


def prefilter_recommendations(recommendations: pd.DataFrame,
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

    return pd.DataFrame(acc_list, columns=['user_id', 'item_id'])


def accept_new_recommendations(choice_model: str,
                               recommendations: pd.DataFrame,
                               demographics: pd.DataFrame,
                               tracks: pd.DataFrame,
                               k: int = 10):
    # TODO implement the other choice models
    # - ranked choice probability
    # - us-centric choice model
    # - non us-centric choice model
    if choice_model == 'random':
        recommendations = choice_model_random(recommendations)
    else:
        raise NotImplementedError('Unknown Choice model!')

    return recommendations
