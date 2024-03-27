from recbole.quick_start import run_recbole
from recbole_wrappers import eval_recbole
from dataset_io import create_datasets
from dataset_io.io_helper import read_predictions
from loop_helpers import top_k_item_ids
import gzip
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
from tqdm import tqdm, tqdm
import shutil

import pdb
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


def get_accepted_songs(rec_df, conditional:bool, inter, rec_repeat:bool, top_number:int, randomly:bool):
    """
    Parameters
    ----------
    rec_df: pandas dataframe of recommendations in the format [uid, tid, rank, score]
    conditional: Boolean Value if accepted songs should be chosen by probability or if all k should be accepted
    inter: interaction dataframe of the form [uid, tid, LEs]
    rec_repeat: Boolean value if we allow for repetition of already interacted songs
    top_number: Number of top k items to be recommended
    randomly: Boolean Value if user should be modeled by randomly choosing one item from the top k

    Returns
    -------
    pandas dataframe of accepted songs according to probability
    """
    acc_list = []

    if randomly: # add to parameters
        # have to loop through users
        p_list = []  # maybe define before calling function
        for i in range(1, top_number+1):
            p_list.append(prob(i, -0.1))
        p_list = p_list / sum(p_list)
        for user_id in rec_df['user_id'].unique():
# TODO: add 0 items prob here
            rank = np.random.choice(range(1, top_number+1), size=1, p=p_list)[0]  # have a look if we can sort of hardcode list
            row = rec_df.loc[(rec_df['user_id'] == user_id) & (rec_df['rank'] == rank)]
            acc_list.append([int(row['user_id']), int(row['item_id'])])
    else:
        for idx, row in tqdm(rec_df.iterrows()):
            if conditional:
                probab = prob(row['rank'], -0.1)
                if random.random() < probab:  # song is accepted according to probability and added to list with estimate of LEs
                    if rec_repeat:  # if we allow for recommendation of already listened to songs
                        # if the song is new we add it to the list of accepted songs
                        if inter.loc[(inter['user_id:token']==int(row['user_id']))&(inter['item_id:token']==int(row['item_id']))].empty:
                            acc_list.append([int(row['user_id']), int(row['item_id'])])
                        # else:
                            # if the song is repeated we increase it's LEs
                            # inter.loc[inter.loc[(inter['uid'] == int(row['user_id'])) &
                            #                    (inter['tid'] == int(row['item_id']))].index, 'LEs'] += 3
                    else:
                        acc_list.append([int(row['user_id']), int(row['item_id'])])
            else:  # all songs are accepted and added to list with estimate of LEs
                if rec_repeat:
                    if inter.loc[(inter['user_id:token'] == int(row['user_id'])) & (inter['item_id:token'] == int(row['item_id']))].empty:
                        acc_list.append([int(row['user_id']), int(row['item_id'])])
                    # else:
                    #    inter.loc[inter.loc[(inter['uid'] == int(row['user_id'])) &
                    #                        (inter['tid'] == int(row['item_id']))].index, 'LEs'] += 3
                else:
                    acc_list.append([int(row['user_id']), int(row['item_id'])])
    acc_df = pd.DataFrame(acc_list, columns=['user_id:token', 'item_id:token'])
    return acc_df

def get_accepted_songs_new(rec_df, strategy: str, filter: str):
    """
    Parameters
    ----------
    rec_df: pandas dataframe of recommendations in the format [uid, tid, rank, score]
    strategy: string - acceptance strategy name

    filter: string, <field>=<value> e.g. track_country=US, the <field> is a column in rec_df
            filter from the natural recommendations (no consumption if empty after filter)

    Returns
    -------
    pandas dataframe of accepted songs according to probability
    """
    # checking if selected acceptance strategy is acceptable
    if strategy not in ['all_topk', 'one_item_from_topk_prob']:
        raise Exception('Unknown acceptance strategy!')

    # defining acceptance strategies
    def all_topk(rec_df):
        res = rec_df
        return res

    def one_item_from_topk_prob(rec_df):
        res = None

        p_list = []
        for i in range(1, len(rec_df)+1):
            p_list.append(prob(i, -0.1))
        p_list = np.array(p_list)
        p_list = p_list / sum(p_list)

        r = np.random.choice(range(1, len(rec_df) + 1), size=1, p=p_list)[0]
        res = rec_df.loc[rec_df['rank'] == r]

        return res

    strategies = {
        'all_topk' : all_topk,
        'one_item_from_topk_prob' : one_item_from_topk_prob
    }
    # fixing selected strategy
    acceptance_strategy = strategies[strategy] # this is a function, give it arguments

    # initializing recommendation filtering
    flt = False
    flt_target = ''
    flt_value = ''

    if '=' in  filter:
        fields = filter.split('=')
        flt_target = fields[0]
        flt_value = fields[1]
        flt = True

    # LET THE ACCEPTANCE COMMENCE
    acc_df = pd.DataFrame({'user_id:token': [], 'item_id:token': []}, dtype=int)

    for user_id in rec_df['user_id'].unique():
        # recommendations to the current user
        current_recs = rec_df.loc[(rec_df['user_id'] == user_id)]
        # filtering the recommendations:
        if flt:
            current_recs = current_recs[current_recs[flt_target] == flt_value]

        # apply the acceptance strategy
        accepted = acceptance_strategy(current_recs)[['user_id', 'item_id']]
        accepted.columns = ['user_id:token', 'item_id:token']

        # append accepted items
        #acc_df = acc_df.append(accepted, ignore_index=True)
        acc_df = pd.concat([acc_df, accepted], ignore_index=True, join='inner')

        # --> proceed to the next user
    return acc_df.reset_index()

    '''
    if strategy == '1_item_from_topk_prob': # add to parameters
        # have to loop through users
        p_list = []  # maybe define before calling function
        for i in range(1, topk+1):
            p_list.append(prob(i, -0.1))
        p_list = p_list / sum(p_list)
        for user_id in rec_df['user_id'].unique():
            # TODO: add 0 items prob here
            rank = np.random.choice(range(1, topk+1), size=1, p=p_list)[0]  # have a look if we can sort of hardcode list
            row = rec_df.loc[(rec_df['user_id'] == user_id) & (rec_df['rank'] == rank)]
            acc_list.append([int(row['user_id']), int(row['item_id'])])
    else:
        for idx, row in tqdm(rec_df.iterrows()):
            if strategy == 'from_topk_prob':
                probab = prob(row['rank'], -0.1)
                if random.random() < probab:  # song is accepted according to probability and added to list with estimate of LEs
                    if rec_repeat:  # if we allow for recommendation of already listened to songs
                        # if the song is new we add it to the list of accepted songs
                        if inter.loc[(inter['user_id:token']==int(row['user_id']))&(inter['item_id:token']==int(row['item_id']))].empty:
                            acc_list.append([int(row['user_id']), int(row['item_id'])])
                        # else:
                            # if the song is repeated we increase it's LEs
                            # inter.loc[inter.loc[(inter['uid'] == int(row['user_id'])) &
                            #                    (inter['tid'] == int(row['item_id']))].index, 'LEs'] += 3
                    else:
                        acc_list.append([int(row['user_id']), int(row['item_id'])])
            else:  # all songs are accepted and added to list with estimate of LEs
                if rec_repeat:
                    if inter.loc[(inter['user_id:token'] == int(row['user_id'])) & (inter['item_id:token'] == int(row['item_id']))].empty:
                        acc_list.append([int(row['user_id']), int(row['item_id'])])
                    # else:
                    #    inter.loc[inter.loc[(inter['uid'] == int(row['user_id'])) &
                    #                        (inter['tid'] == int(row['item_id']))].index, 'LEs'] += 3
                else:
                    acc_list.append([int(row['user_id']), int(row['item_id'])])
    acc_df = pd.DataFrame(acc_list, columns=['user_id:token', 'item_id:token'])
    return acc_df
    '''

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
    history_file = dataset_path / 'history.npy'
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
    np.save(history_file, data)


@cli('dataset', type=str, help='dataset name to evaluate')
@cli('-m', '--model', type=str, help='RecBole model used')
@cli('-t', '--loops', type=int, help='Number of loops')
@cli('-k', '--top_number', type=int, help='Number of recommended items')
@cli('-c', '--count', type=int, help='Only change for continuing previous experiment. Last iteration + 1 and < loops')
@cli('-cc', '--country', type=str, help='Only used when --freeze_country = True, country code of the country you want '
                                        'to freeze')
@cli('-p', '--control_perc', type=float, help='Percentage of users which should belong to the control group, '
                                              'float between 0 and 1')
@cli('--freeze_country', action=argparse.BooleanOptionalAction, help='Music is not recommended to specified country, '
                                                                     'specify country with -cc')
@cli('--control_group', action=argparse.BooleanOptionalAction, help='Use a fixed control group of random users who do '
                                                                    'not get recommendations')
@cli('--strategy', type=str, help='Track acceptance strategy')
@cli('--acceptance_filter', type=str, help='Additional filter for acceptance strategy, e.g. track_country == US')
@cli('--rec_repeat', action=argparse.BooleanOptionalAction, help='Allowing repeated recommendation')
@cli('--delete', action=argparse.BooleanOptionalAction, help='Delete redundant files')
def loop(dataset: str, model: str = 'MultiVAE', loops: int = 20, top_number: int = 10, freeze_country: bool = False,
         country: str = 'FI', control_group: bool = False, control_perc: float = 0.10, strategy: str = 'all_topk',
         acceptance_filter: str = '', rec_repeat: bool = False, delete: bool = True, count: int = 1):
    """
    Function to automatically run a --sequential-- recommendation simulation on a given dataset.

    Parameters
    ----------
    dataset: str, Dataset name to evaluate, if you continue previous experiment it needs to be the name of the last
             dataset created
    model: str, Model to use for evaluation
    loops: int, Number of iterations to perform
    top_number: int, number of top items to consider
    freeze_country: bool, If true, songs are not recommended to users coming from the country defined in country.
    country: str, Country code of country for which users should not get new recommendations, only used when
             freeze_country is True
    control_group: bool, If True, then a randomly selected subgroup of users does not receive recommendations throughout
                   the loop iterations
    control_perc: float, between 0 and 1, defines the percentage of users selected for the control_group, only used if
                  control_group is True
    rec_repeat: bool, True if songs should be allowed to be repeated, False if only new songs should be recommended
    delete: bool, True if redundant files should be deleted while running (does not affect original files and new inter
            and top_k.text files), False if all files should be saved while running, Default = True
    count:  Only change when you continue a previously started experiment, needs to be 'count of the last iteration' + 1
            and needs to be smaller than loops

    Returns
    -------
    Saves new .inter, model_top_k.txt, model_accepted_songs.txt files to use for further evaluation
    """
    # count = 1  # count to keep track of loops
    cont_group = None
    if freeze_country:
        fu = pd.read_csv(dataset_root / dataset / 'demo.txt.gz', sep='\t', header=None,
                         names=['country', 'age', 'gender', 'time'])
        c_freeze = fu.copy().loc[fu['country'] == country] ### COPY ??? ###
        c_ind = list(c_freeze.index)
        del fu
    while count <= loops:
        print(f'This is loop {count}')
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")  # get current time at start of the loop
        print("Current Time =", current_time)
        # dataset has to be in recbole format already, if not we need to create it here
        # print('Creating the .inter file from the given dataset.')
        # os.system(f'python dataset_io/create_datasets.py {dataset}')

        print('Running recbole on the new .inter file.')


        ###############Aaaagh!

        #run_recbole(model=model, dataset=dataset, config_file_list=['recbole_config.yaml'])

        ###############

        print('Accessing the most recent model file.')
        list_of_models = glob.glob('saved/*.pth')  # getting list of possible models and accessing the most recent one
        latest_file = max(list_of_models, key=os.path.getctime)
        print(f'Model used for evaluation : {latest_file}')

        print('Evaluating the model and saving scores.')
        os.system(f'python recbole_wrappers/eval_recbole.py {latest_file}')

        print('Scores are being read.')
        scores, _ = read_predictions(dataset_root / dataset / f'{model}.npz')
        # del actual
        if not rec_repeat:  # if recs should not be repeated the scores are multiplied with the inverse history to only recommend new songs
            scores = np.nan_to_num(scores, neginf=-1)
            get_history(dataset)
            hist = np.load(dataset_root / dataset / 'history.npy')  # load history file
            inv_hist = (~hist.astype(bool)).astype(int)  # inverse interaction history
            scores = scores * inv_hist
            del hist
            del inv_hist
        print(f'Top {top_number} items are being calculated.')
        top_k = top_k_item_ids(scores, top_number)  # get top k item ids
        top_k.to_csv(dataset_root / dataset / f'{model}_top_k.txt', header=None, sep='\t', index = False)  # save them as a txt file
        print(f'Top k items saved as {model}_top_k.txt')
        del scores

        print('Accepted songs are calculated and added to the inter file')
        fi = pd.read_csv(dataset_root / dataset / f'{dataset}.inter', sep='\t', header = 0, names=['user_id:token', 'item_id:token'])
        # load the interaction file
        # accepted_songs = get_accepted_songs(top_k, conditional, fi.copy(), rec_repeat, top_number, randomly)  # calculate accepted songs and LEs
        accepted_songs = get_accepted_songs_new(top_k, strategy, acceptance_filter)
        if freeze_country:
            accepted_songs = accepted_songs[~accepted_songs['user_id:token'].isin(c_ind)]
        if control_group:
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
        df_new_recs = pd.concat([fi.copy(), accepted_songs], ignore_index=True)
        # add new songs to old interactions

        pdb.set_trace()
        new_df = df_new_recs.copy().sort_values(['user_id:token', 'item_id:token'])  # sort interactions
        new_df.reset_index(drop=True)
        del fi
        del df_new_recs
        del top_k
        df = pd.read_csv(dataset_root / dataset / f'{dataset}.train.inter', sep='\t', header = 0,
                         names=['user_id:token', 'item_id:token'])
        df_train_recs = pd.concat([df.copy(), accepted_songs], ignore_index=True)
        train_df = df_train_recs.copy().sort_values(['user_id:token', 'item_id:token'])  # sort interactions
        train_df.reset_index(drop=True)
        del df
        del df_train_recs
        del accepted_songs
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
            new_df.to_csv(dataset_root / f'{dataset}_{count}' / f'{dataset}_{count}.inter', sep='\t',
                          header=['user_id:token', 'item_id:token'], index=False)
            train_df.to_csv(dataset_root / f'{dataset}_{count}' / f'{dataset}_{count}.train.inter', sep='\t',
                            header=['user_id:token', 'item_id:token'], index=False)
            shutil.copy(dataset_root / f'{dataset}' / f'{dataset}.validate.inter',
                        dataset_root / f'{dataset}_{count}' / f'{dataset}_{count}.validate.inter')
            shutil.copy(dataset_root / f'{dataset}' / f'{dataset}.test.inter',
                        dataset_root / f'{dataset}_{count}' / f'{dataset}_{count}.test.inter')
            print(f'New inter file was saved under {dataset}_{count}/{dataset}_{count}.inter .')
            dataset = f'{dataset}_{count}'
            del new_df
        else:
            x = dataset.rfind('_')
            os.makedirs(dataset_root / f'{dataset[:x]}_{count}')
            new_df.to_csv(dataset_root / f'{dataset[:x]}_{count}' / f'{dataset[:x]}_{count}.inter', sep='\t',
                          header=['user_id:token', 'item_id:token'], index=False)
            train_df.to_csv(dataset_root / f'{dataset[:x]}_{count}' / f'{dataset[:x]}_{count}.train.inter', sep='\t',
                            header=['user_id:token', 'item_id:token'], index=False)
            shutil.copy(dataset_root / f'{dataset}' / f'{dataset}.validate.inter',
                        dataset_root / f'{dataset[:x]}_{count}' / f'{dataset[:x]}_{count}.validate.inter')
            shutil.copy(dataset_root / f'{dataset}' / f'{dataset}.test.inter',
                        dataset_root / f'{dataset[:x]}_{count}' / f'{dataset[:x]}_{count}.test.inter')
            print(f'New inter file was saved under {dataset[:x]}_{count}/{dataset[:x]}_{count}.inter .')
            dataset = f'{dataset[:x]}_{count}'
            del new_df
        count += 1  # increase iteration counter


if __name__ == '__main__':
    argh.dispatch_command(loop)

