import gc

from recbole.quick_start import run_recbole
from recbole_wrappers import eval_recbole
from recbole.config import Config
from dataset_io import create_datasets
from dataset_io.io_helper import read_predictions
from loop_helpers import top_k_item_ids
import gzip
import torch
import random
from pathlib import Path
from datetime import datetime
import argh
import argparse
import os
import glob
import numpy as np
import pandas as pd
from argh import arg as cli
from tqdm import tqdm
import shutil
from debias.falsepositivecorrection import FalsePositiveCorrection

# from dataset_io.create_datasets import dataset_root
dataset_root = Path('data')
random.seed(239476524)


def prob(rank, alpha):
    """
    Parameters
    ----------
    rank: rank of the recommended item
    alpha: negative value for controlling probability

    Returns
    -------
    probability of user accepting the song
    """
    # Probability is calculated according to 'Mansoury, M., Abdollahpouri, H., Pechenizkiy, M., Mobasher, B., and Burke, R.
    # Feedback loop and bias amplification in recommender systems. In Proceedings of the 29th ACM
    # international conference on information & knowledge management (2020), pp. 2145–2148.'
    return np.exp(alpha*rank)


def item_response(user_id, track_id, inter):
    """
    Function to calculate the LEs for a recommended item according to the user and item interactions.
    Parameters
    ----------
    user_id: int, id of the user
    track_id: int, id of the song
    inter: pandas dataframe, interaction of all users and songs

    Returns
    -------
    Number of LEs for new song for user

    """
    # Formula is taken from 'Mansoury, M., Abdollahpouri, H., Pechenizkiy, M., Mobasher, B., and Burke, R.
    # Feedback loop and bias amplification in recommender systems. In Proceedings of the 29th ACM
    # international conference on information & knowledge management (2020), pp. 2145–2148.' and adapted to the data
    user_in = inter.loc[inter['uid'] == user_id]
    mean_s_u = user_in['LEs'].mean()  # average of LEs in users profile
    std_s_u = user_in['LEs'].std()  # standard deviation of LEs of user
    track_in = inter.loc[inter['tid'] == track_id]
    mean_s_i = track_in['LEs'].mean()  # average LEs of song
    noise = np.random.normal()  # random noise
    omega = mean_s_u + (std_s_u/10 * mean_s_i) + noise
    return int(np.round(omega))


def get_accepted_songs(rec_df, conditional: bool, inter, rec_repeat: bool, top_number: int, randomly: bool, valid_track_indices=None, accept_all=False, at_most_one=False):
    """
    Parameters
    ----------
    rec_df: pandas dataframe of recommendations in the format [uid, tid, rank, score]
    conditional: Boolean Value if accepted songs should be chosen by probability or if all k should be accepted
    inter: interaction dataframe of the form [uid, tid, LEs]
    rec_repeat: Boolean value if we allow for repetition of already interacted songs
    top_number: Number of top k items to be recommended
    randomly: Boolean Value if user should be modeled by randomly choosing one item from the top k
    valid_track_indices: Is a python list on the items to e.g. only accept US songs (only accepts the intersection)
    accept_all: if valid_track_indices is true, then ignore smarter modeling but just accept everything recommended to the user from the df
    Returns
    -------
    pandas dataframe of accepted songs according to probability
    """

    # Work on a copy of the dataframe
    filtered_df = rec_df.copy()

    # Filter the recommendation dataframe based on valid_track_indices
    if valid_track_indices is not None:
        filtered_df = filtered_df[filtered_df['item_id'].isin(valid_track_indices)]

    # Accept all recommendations if accept_all is True
    if accept_all:
        return filtered_df[['user_id', 'item_id']].rename(columns={'user_id': 'user_id:token', 'item_id': 'item_id:token'})

    acc_list = []

    if randomly:
        p_list = [prob(i, -0.1) for i in range(1, top_number + 1)]
        p_list = np.array(p_list) / sum(p_list)  # Normalize probabilities

        for user_id in filtered_df['user_id'].unique():
            rank = np.random.choice(range(1, top_number + 1), p=p_list)
            row = filtered_df.loc[(filtered_df['user_id'] == user_id) & (filtered_df['rank'] == rank)]
            acc_list.append([int(row['user_id']), int(row['item_id'])])
    else:
        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
            if conditional:
                probab = prob(row['rank'], -0.1)
                if random.random() < probab:
                    if rec_repeat or inter.loc[(inter['user_id:token'] == int(row['user_id'])) & (inter['item_id:token'] == int(row['item_id']))].empty:
                        acc_list.append([int(row['user_id']), int(row['item_id'])])
            else:
                if rec_repeat or inter.loc[(inter['user_id:token'] == int(row['user_id'])) & (inter['item_id:token'] == int(row['item_id']))].empty:
                    acc_list.append([int(row['user_id']), int(row['item_id'])])

    acc_df = pd.DataFrame(acc_list, columns=['user_id:token', 'item_id:token'])
    if at_most_one:
        acc_df = acc_df.groupby('user_id:token').first().reset_index()
    return acc_df

def get_accepted_songs_only_own_country(rec_df, conditional, inter, rec_repeat, top_number, randomly, country_valid_tracks, users_to_country, accept_all):
    filtered_df = rec_df.copy()
    users_country_df = pd.DataFrame(list(users_to_country.items()), columns=['user_id', 'country'])
    merged_df = pd.merge(filtered_df, users_country_df, on='user_id')
    def is_valid_track(row):
        return row['item_id'] in country_valid_tracks.get(row['country'], [])
    
    filtered_df = merged_df[merged_df.apply(is_valid_track, axis=1)]
    filtered_df = filtered_df[["user_id", "item_id", "rank", "score"]]

    # Accept all recommendations if accept_all is True
    if accept_all:
        return filtered_df[['user_id', 'item_id']].rename(columns={'user_id': 'user_id:token', 'item_id': 'item_id:token'})

    acc_list = []

    if randomly:
        p_list = [prob(i, -0.1) for i in range(1, top_number + 1)]
        p_list = np.array(p_list) / sum(p_list)  # Normalize probabilities

        for user_id in filtered_df['user_id'].unique():
            rank = np.random.choice(range(1, top_number + 1), p=p_list)
            row = filtered_df.loc[(filtered_df['user_id'] == user_id) & (filtered_df['rank'] == rank)]
            acc_list.append([int(row['user_id']), int(row['item_id'])])
    else:
        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
            if conditional:
                probab = prob(row['rank'], -0.1)
                if random.random() < probab:
                    if rec_repeat or inter.loc[(inter['user_id:token'] == int(row['user_id'])) & (inter['item_id:token'] == int(row['item_id']))].empty:
                        acc_list.append([int(row['user_id']), int(row['item_id'])])
            else:
                if rec_repeat or inter.loc[(inter['user_id:token'] == int(row['user_id'])) & (inter['item_id:token'] == int(row['item_id']))].empty:
                    acc_list.append([int(row['user_id']), int(row['item_id'])])

    acc_df = pd.DataFrame(acc_list, columns=['user_id:token', 'item_id:token'])
    return acc_df

def get_history(dataset: str):
    """
    Parameters
    ----------
    dataset: name of the dataset for which the history file should be created

    Returns
    -------
    Creates history.npy file for the dataset and saves it.
    """
    dataset_path = dataset_root / dataset

    inter_file = dataset_path / f'{dataset}.inter'
    #history_file = dataset_path / 'history.npy'
    dataset_pandas = pd.read_csv(inter_file, sep='\t', header = 0, names=['user_id:token', 'item_id:token'])

    data = np.zeros((
        len(pd.unique(dataset_pandas['user_id:token'])),
        len(pd.unique(dataset_pandas['item_id:token']))
    ),
        dtype=np.uint8
    )

    for index, line in tqdm(dataset_pandas.iterrows(), desc=f'creating history file', total=len(dataset_pandas), smoothing=0):
        user = int(line['user_id:token'])
        item = int(line['item_id:token'])
        data[user, item] = 1
    return data

def find_last_iteration(dataset_root, base_dataset_name):
    existing_dirs = glob.glob(f"{dataset_root}/{base_dataset_name}_*")
    if not existing_dirs:
        return None

    last_iteration = max(existing_dirs, key=lambda x: int(x.rsplit('_', 1)[-1]))
    return last_iteration

@cli('dataset', type=str, help='dataset name to evaluate. This is taken to be a subfolder of data/')
@cli('-m', '--model', type=str, help='RecBole model used')
@cli('-t', '--loops', type=int, help='Number of final loops that should be in the output (e.g. if resume is used and dataset_5 exists then --loops is set to 7, this script will run 2 times)')
@cli('-k', '--top_number', type=int, help='Number of recommended items')
@cli('-c', '--count', type=int, help='Only change for continuing previous experiment. Last iteration + 1 and < loops')
@cli('-cc', '--country', type=str, help='Only used when --freeze_country = True or --only-from-country = True, country code of the country you want '
                                        'to freeze')
@cli('-ccc', '--control-group-country', type=str, help='Only used when control group is true - this should be the country code')
@cli('-p', '--control_perc', type=float, help='Percentage of users which should belong to the control group, '
                                              'float between 0 and 1')
@cli('--freeze_country', action=argparse.BooleanOptionalAction, help='Music is not recommended to specified country, '
                                                                     'specify country with -cc')
@cli('--only-from-country', action=argparse.BooleanOptionalAction, help='Music is only accepted from a specified country, '
                                                                     'specify country with -cc')
@cli('--not-from-country', action=argparse.BooleanOptionalAction, help='Inverts only from country and only recomments what does not stem from a certain country'
                                                                     'specify country with -cc')
@cli('--control_group', action=argparse.BooleanOptionalAction, help='Use a fixed control group of random users who do '
                                                                    'not get recommendations')
@cli('--conditional', action=argparse.BooleanOptionalAction, help='Using acceptance strategy')
@cli('--rec_repeat', action=argparse.BooleanOptionalAction, help='Allowing repeated recommendation')
@cli('--accept-all', action=argparse.BooleanOptionalAction, help='Modifies the choice-model to accept all recommendations which are relevant'
                                                                    '(posiibly filtered by --only-from-country)')
@cli('--randomly', action=argparse.BooleanOptionalAction, help='User is modeled by randomly choosing one item from the '
                                                               'top k')
@cli('--delete', action=argparse.BooleanOptionalAction, help='Delete redundant files')
@cli('--resume', action=argparse.BooleanOptionalAction, help='Resume from the last checkpoint if available looks for <dataset_name>_<iteration_nr>')
@cli('--one-loop', action=argparse.BooleanOptionalAction, help='Only run for one more iteration, only use with --resume to compute the target loop number')
@cli('--only-own-country', action=argparse.BooleanOptionalAction, help='Users will only accept songs from their respective country')
@cli('--use-fpc', action=argparse.BooleanOptionalAction, help='Use the false-positive-correction debiasing-method')
@cli('--at-most-one', action=argparse.BooleanOptionalAction, help='Accept at most one recommendation (filters afterwards - not for onlyOwnCountry!)')
def loop(dataset: str, model: str = 'MultiVAE', loops: int = 20, top_number: int = 10, freeze_country: bool = False, only_from_country: bool = False, not_from_country: bool = False,
         country: str = 'FI', control_group_country=None, control_group: bool = False, control_perc: float = 0.10, conditional: bool = False,
         rec_repeat: bool = False, delete: bool = True, accept_all: bool=False, randomly: bool = False, count: int = 1, resume=False, one_loop=False, only_own_country = False, use_fpc = False, at_most_one=False):
    """
    Function to automatically run a sequential recommendation simulation on a given dataset.
    Saves new .inter, model_top_k.txt, model_accepted_songs.txt files to use for further evaluation
    """
    print("Applied Parameters:")
    print(f"  - Dataset: {dataset}")
    print(f"  - Model: {model}")
    print(f"  - Number of Loops: {loops}")
    print(f"  - Top Number: {top_number}")
    print(f"  - Freeze Country: {freeze_country}")
    if freeze_country:
        print(f"    - Frozen Country Code: {country}")
    print(f"  - Only From Country: {only_from_country}")
    print(f"  - Not From Country: {not_from_country}")
    if only_from_country or not_from_country:
        print(f"    - Only/Not From Country Code: {country}")
    print(f"  - Control Group: {control_group}")
    if control_group:
        print(f"    - Control Group Country Code: {control_group_country}")
        print(f"    - Control Group Percentage: {control_perc}")
    print(f"  - Conditional Acceptance: {conditional}")
    print(f"  - Allow Recommendation Repeats: {rec_repeat}")
    print(f"  - Delete Redundant Files: {delete}")
    print(f"  - Accept All Relevant Recommendations: {accept_all}")
    print(f"  - Randomly Choose One Item: {randomly}")
    print(f"  - Count: {count}")
    print(f"  - Resume: {resume}")
    print(f"  - One loop only: {one_loop}")
    print(f"  - One own country: {only_own_country}")
    print(f"  - At most one: {at_most_one}")
    assert not (only_from_country and not_from_country)

    if use_fpc:
        all_accept_songs_path = dataset_root / dataset / 'fpc_all_accepted_songs.buffer.npy'
        all_top_k_path = dataset_root / dataset / 'fpc_all_top_K.buffer.npy'

    if resume:
        last_iteration_dir = find_last_iteration(dataset_root, dataset)

        if last_iteration_dir:
            print(f"Resuming from checkpoint: {last_iteration_dir}")
            count = int(last_iteration_dir.rsplit('_', 1)[-1]) + 1
            dataset = last_iteration_dir.replace(str(dataset_root)+"\\", "", 1).replace(str(dataset_root)+"/", "", 1) # accounts for Linux and Windows paths
            if one_loop:
                loops = count
        else:
            print("No checkpoint found. Starting from the beginning.")
            if one_loop:
                loops = 1

    cont_group = None
    if freeze_country:
        fu = pd.read_csv(dataset_root / dataset / 'demo.txt.gz', sep='\t', header=None,
                         names=['country', 'age', 'gender', 'time'])
        c_freeze = fu.copy().loc[fu['country'] == country]
        c_ind = list(c_freeze.index)
        del fu

    if only_from_country or not_from_country:
        fu = pd.read_csv(dataset_root / dataset / 'tracks.txt.gz', sep='\t', header=None,
                         names=['title', 'band', 'country', 'time'])
        if only_from_country:
            c_freeze = fu.copy().loc[fu['country'] == country]
        else:
            c_freeze = fu.copy().loc[fu['country'] != country]
        valid_tracks_indices = list(c_freeze.index)
        del fu
    elif only_own_country:
        tracks_df = pd.read_csv(dataset_root / dataset / 'tracks.txt.gz', sep='\t', header=None,
                         names=['title', 'band', 'country', 'time'])
        users_df = pd.read_csv(dataset_root / dataset / 'demo.txt.gz', sep='\t', header=None,
                         names=['country', 'age', 'gender', 'time'])
        users_to_country = dict(zip(users_df.index, users_df['country']))
        countries = set(users_df['country'].values)
        country_valid_tracks = {country:tracks_df.copy().loc[tracks_df["country"]==country].index for country in countries}
    else: 
        valid_tracks_indices = None

    while count <= loops:
        print(f'This is loop {count}')
        start_time = datetime.now()

        current_time = start_time.strftime("%H:%M:%S")  # get current time at start of the loop
        print("Current Time =", current_time)
        # dataset has to be in recbole format already, if not we need to create it here
        # print('Creating the .inter file from the given dataset.')
        # os.system(f'python dataset_io/create_datasets.py {dataset}')

        config_filename = "recbole_config.yaml" if model != "ItemKNN" else "recbole_config_knn.yaml"

        config = Config(model=model, dataset=dataset, config_file_list=[config_filename])
        print('Running recbole on the new .inter file.')
        run_recbole(model=model, dataset=dataset, config_file_list=[config_filename])

        print('Accessing the most recent model file.')
        # Martin: Only find latest model of given model class
        list_of_models = list(filter(lambda x: x[6:].startswith(model), glob.glob('saved/*.pth')))  # getting list of possible models and accessing the most recent one
        latest_file = max(list_of_models, key=os.path.getctime)
        print(f'Model used for evaluation : {latest_file}')

        print('Evaluating the model and saving scores.')
        # bad: os.system(f'python recbole_wrappers/eval_recbole.py {latest_file}')
        eval_recbole.evaluate_models([latest_file])

        print('Scores are being read.')
        scores, actual = read_predictions(dataset_root / dataset / f'{model}.npz')
        if use_fpc:
            if count == 1:
                print("Creating empty accepted_histories for FPC")
                accepted_songs, top_k = None , None
                all_top_K = np.zeros((1, scores.shape[0], top_number), dtype=np.int8)
                all_accept_songs = np.zeros((1, scores.shape[0], top_number), dtype=np.int8)
            else:
                print("Loading accepted_histories for FPC")
                all_top_K = np.load(all_top_k_path)
                all_accept_songs = np.load(all_accept_songs_path)

                if count == 2:
                    adjust_dataset = "_".join(dataset.split("_")[:-1])
                else:
                    adjust_dataset = "_".join(dataset.split("_")[:-1]) + "_" + str(count - 2)

                last_iter_accept_songs = pd.read_csv(dataset_root / adjust_dataset / f'{model}_all_accepted_songs.txt', sep="\t")
                last_iter_top_K = pd.read_csv(dataset_root / adjust_dataset / f'{model}_top_k.txt', sep="\t", header=None).values[:,1]

                scores, all_accept_songs, all_top_K = FalsePositiveCorrection(scores, count, top_number, last_iter_accept_songs, last_iter_top_K, all_accept_songs, all_top_K)
            
            print("Storing all_top_K and all_accept_songs for FPC")
            np.save(all_top_k_path, all_top_K)
            np.save(all_accept_songs_path, all_accept_songs)
            del all_accept_songs
            del all_top_K



        del actual
        if not rec_repeat:
            flat_scores = scores.flatten()
            #if len(flat_scores[(flat_scores>-10) & (flat_scores <0)]) > 0: Will throw an error for BPR/MultiVAE since can be neg
            #    raise RuntimeError("Got an invalid score (should be >=0)")
            
            min_score = flat_scores[flat_scores != -np.inf].min()
            del flat_scores
            # if recs should not be repeated the scores are multiplied with the inverse history to only recommend new songs
            scores = np.nan_to_num(scores, neginf=min_score) # Martin: not -1 any more since BPR can also give negative Predictions
            hist = get_history(dataset)
            scores[hist==1] = min_score # set every repeat to the minimal score
            del hist
        print(f'Top {top_number} items are being calculated.')
        top_k = top_k_item_ids(scores, top_number)  # get top k item ids
        top_k.to_csv(dataset_root / dataset / f'{model}_top_k.txt', header=None, sep='\t', index = False)  # save them as a txt file
        print(f'Top k items saved as {model}_top_k.txt')

        
        del scores

        print('Accepted songs are calculated and added to the inter file')
        fi = pd.read_csv(dataset_root / dataset / f'{dataset}.inter', sep='\t', header = 0, names=['user_id:token', 'item_id:token'])
        # load the interaction file
        if only_own_country:
            accepted_songs = get_accepted_songs_only_own_country(top_k, conditional, fi.copy(), rec_repeat, top_number, randomly, country_valid_tracks, users_to_country, accept_all)  # calculate accepted songs and LEs
        else:
            accepted_songs = get_accepted_songs(top_k, conditional, fi.copy(), rec_repeat, top_number, randomly, valid_tracks_indices, accept_all, at_most_one)  # calculate accepted songs and LEs

        if use_fpc:
            accepted_songs.to_csv(dataset_root / dataset / f'{model}_all_accepted_songs.txt',
                                header=['user_id:token', 'item_id:token'], sep='\t', index = False)  # save them as a txt file
        
        if freeze_country:
            accepted_songs = accepted_songs[~accepted_songs['user_id:token'].isin(c_ind)]
        if control_group:
            if cont_group is None: # we have to get a list of all the users of the specific control group country
                if control_group_country is not None:
                    print(f"Using {control_group_country} as a control group")
                    fu = pd.read_csv(dataset_root / dataset / 'demo.txt.gz', sep='\t', header=None,
                            names=['country', 'age', 'gender', 'time'])
                    c_freeze = fu.copy().loc[fu['country'] == control_group_country]
                    cont_group = list(c_freeze.index)
                    with open(dataset_root / dataset / 'control_group.txt', 'w') as f:
                        for line in cont_group:
                            f.write(f"{line}\n")
                    del fu
                else:
                    print(f"Using control_group.txt as a control group")
                    if count == 1:
                        if os.path.exists(dataset_root / dataset / 'control_group.txt'):
                            with open(dataset_root / dataset / 'control_group.txt', 'r') as file:
                                cont_group = [int(line.strip()) for line in file]
                        else:
                            uids = fi['user_id:token'].unique()
                            cont_group = random.sample(list(uids), int(len(uids) * control_perc))
                            with open(dataset_root / dataset / 'control_group.txt', 'w') as f:
                                for line in cont_group:
                                    f.write(f"{line}\n")
                    else:
                        if cont_group is None:
                            try:
                                if os.path.exists(dataset_root / dataset / 'control_group.txt'):
                                    with open(dataset_root / dataset / 'control_group.txt', 'r') as file:
                                        cont_group = [int(line.strip()) for line in file]
                                else:
                                    raise FileNotFoundError(
                                        f"The file {dataset}/control_group.txt does not exist. Create a new control_group.txt "
                                        f"file to continue training or move existing one to correct folder.")
                            except FileNotFoundError as ex:
                                raise ex
                accepted_songs = accepted_songs[~accepted_songs['user_id:token'].isin(cont_group)]
        accepted_songs.to_csv(dataset_root / dataset / f'{model}_accepted_songs.txt',
                              header=['user_id:token', 'item_id:token'], sep='\t', index = False)  # save them as a txt file

        """Max: Add a part of the accepted songs to the validation set"""
        # Get validation ratio from config
        validation_ratio: float = config['eval_args']['split']['RS'][1]
        # Split of from the correct ratio from accepted songs
        accepted_songs_validation = accepted_songs.sample(frac=validation_ratio, random_state=1)
        accepted_songs_train = accepted_songs.drop(accepted_songs_validation.index)

        df_new = pd.concat([fi.copy(), accepted_songs], ignore_index=True)
        df_new = df_new.sort_values(['user_id:token', 'item_id:token']).reset_index(drop=True)

        df_train = pd.read_csv(dataset_root / dataset / f'{dataset}.train.inter', sep='\t', header = 0,
                               names=['user_id:token', 'item_id:token'])
        df_new_train = pd.concat([df_train.copy(), accepted_songs_train], ignore_index=True)
        df_new_train = df_new_train.sort_values(['user_id:token', 'item_id:token']).reset_index(drop=True)  # sort interactions


        df_val = pd.read_csv(dataset_root / dataset / f'{dataset}.validate.inter', sep='\t', header = 0,
                                 names=['user_id:token', 'item_id:token'])
        df_new_val = pd.concat([df_val.copy(), accepted_songs_validation], ignore_index=True)
        df_new_val = df_new_val.sort_values(['user_id:token', 'item_id:token']).reset_index(drop=True)  # sort interactions

        if delete:  # delete redundant files for storage optimization
            # delete history file
            if os.path.exists(dataset_root / dataset / "history.npy"):
                os.remove(dataset_root / dataset / "history.npy")
            else:
                print(f"The file {dataset}/history.npy does not exist")
            # delete the scores of the recommendations
            if os.path.exists(dataset_root / dataset / f'{model}.npz'):
                os.remove(dataset_root / dataset / f'{model}.npz')
            else:
                print(f"The file {dataset}/{model}.npz does not exist")
            # delete the .inter file of the interactions
            # if os.path.exists(dataset_root / dataset / f'{dataset}.inter'):
            #    os.remove(dataset_root / dataset / f'{dataset}.inter')
            #else:
            #    print(f"The file {dataset}/{dataset}.inter does not exist")
            # delete the saved model and dataloaders
            list_of_models.sort(key=lambda x: os.path.getmtime(x))
            for file in list_of_models[-3:]:
                if os.path.exists(file):
                    os.remove(file)
                else:
                    print(f"The file {file} does not exist")
        # create new dataset folder and inter.txt.gz file for next iteration, number of file indicates iteration
        if count == 1:
            os.makedirs(dataset_root / f'{dataset}_{count}')
            df_new.to_csv(dataset_root / f'{dataset}_{count}' / f'{dataset}_{count}.inter', sep='\t',
                          header=['user_id:token', 'item_id:token'], index=False)
            df_new_train.to_csv(dataset_root / f'{dataset}_{count}' / f'{dataset}_{count}.train.inter', sep='\t',
                            header=['user_id:token', 'item_id:token'], index=False)
            df_new_val.to_csv(dataset_root / f'{dataset}_{count}' / f'{dataset}_{count}.validate.inter', sep='\t',
                            header=['user_id:token', 'item_id:token'], index=False)
            shutil.copy(dataset_root / f'{dataset}' / f'{dataset}.test.inter',
                        dataset_root / f'{dataset}_{count}' / f'{dataset}_{count}.test.inter')
            shutil.copy(dataset_root / f'{dataset}' / f'demo.txt.gz',
                        dataset_root / f'{dataset}_{count}' / f'demo.txt.gz')
            shutil.copy(dataset_root / f'{dataset}' / f'tracks.txt.gz',
                        dataset_root / f'{dataset}_{count}' / f'tracks.txt.gz')
            print(f'New inter file was saved under {dataset}_{count}/{dataset}_{count}.inter .')
            dataset = f'{dataset}_{count}'
        else:
            x = dataset.rfind('_')
            os.makedirs(dataset_root / f'{dataset[:x]}_{count}')
            df_new.to_csv(dataset_root / f'{dataset[:x]}_{count}' / f'{dataset[:x]}_{count}.inter', sep='\t',
                          header=['user_id:token', 'item_id:token'], index=False)
            df_new_train.to_csv(dataset_root / f'{dataset[:x]}_{count}' / f'{dataset[:x]}_{count}.train.inter', sep='\t',
                            header=['user_id:token', 'item_id:token'], index=False)
            df_new_val.to_csv(dataset_root / f'{dataset[:x]}_{count}' / f'{dataset[:x]}_{count}.validate.inter', sep='\t',
                            header=['user_id:token', 'item_id:token'], index=False)
            shutil.copy(dataset_root / f'{dataset}' / f'{dataset}.test.inter',
                        dataset_root / f'{dataset[:x]}_{count}' / f'{dataset[:x]}_{count}.test.inter')
            shutil.copy(dataset_root / f'{dataset}' / f'demo.txt.gz',
                        dataset_root / f'{dataset[:x]}_{count}' / f'demo.txt.gz')
            shutil.copy(dataset_root / f'{dataset}' / f'tracks.txt.gz',
                        dataset_root / f'{dataset[:x]}_{count}' / f'tracks.txt.gz')
            print(f'New inter file was saved under {dataset[:x]}_{count}/{dataset[:x]}_{count}.inter .')
            dataset = f'{dataset[:x]}_{count}'

        print(f'Time elapsed for loop {count}: {datetime.now() - start_time}')
        # By Max: Attempts at keeping the loop fast and not clog up with whatever is causing the weird behaviour
        torch.cuda.empty_cache()
        gc.collect()

        count += 1  # increase iteration counter


if __name__ == '__main__':
    argh.dispatch_command(loop)

