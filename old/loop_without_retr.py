import math
import sys
import os
import shutil

import argh
import argparse
import bottleneck as bn
import numpy as np
import torch
import pandas as pd
import random
import glob
from argh import arg as cli
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_scores
from recbole.quick_start import run_recbole
from recbole.data import (
    create_dataset,
    data_preparation,
    save_split_dataloaders,
    load_split_dataloaders,
)
from recbole.config import Config
from tqdm import tqdm, trange
from dataset_io.create_datasets import dataset_root
from dataset_io.io_helper import read_predictions
from loop_helpers import top_k_item_ids
from recbole_wrappers.eval_recbole import eval_loaded_model  # binary_NDCG, _get_ids, _get_matrices,


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
    #std_s_u = user_in['LEs'].std()  # standard deviation of LEs of user
    #track_in = inter.loc[inter['tid'] == track_id]
    #mean_s_i = track_in['LEs'].mean()  # average LEs of song
    #noise = np.random.normal()  # random noise
    omega = mean_s_u # + (std_s_u/10 * mean_s_i) + noise
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
@cli('--conditional', action=argparse.BooleanOptionalAction, help='Using acceptance strategy')
@cli('--rec_repeat', action=argparse.BooleanOptionalAction, help='Allowing repeated recommendation')
@cli('--rec_repeat', action=argparse.BooleanOptionalAction, help='Allowing repeated recommendation')
@cli('--delete', action=argparse.BooleanOptionalAction, help='Delete redundant files')
def loop_without_retraining(dataset: str, model: str = 'MultiVAE', loops: int = 20, top_number: int = 10,
                            freeze_country: bool = False, country: str = 'FI', control_group: bool = False,
                            control_perc: float = 0.10, conditional: bool = False, rec_repeat: bool = False,
                            delete: bool = True, randomly: bool = False, count: int = 1, model_path: str = None):
    """
    Function to automatically run a sequential recommendation simulation on a given dataset without retraining the model.

    Parameters
    ----------
    dataset: str, Dataset name to evaluate
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
    conditional: bool, True if accepted songs should be accepted according to probability, False if all of them should be accepted
    rec_repeat: bool, True if songs should be allowed to be repeated, False if only new songs should be recommended
    delete: bool, True if redundant files should be deleted while running (does not affect original files and new inter
            and top_k.text files), False if all files should be saved while running, Default = True
    count:  int, Only change when you continue a previously started experiment, needs to be
            'count of the last iteration' + 1 and needs to be smaller than loops
    model_path: str, Only change when you continue a previously started experiment, Path to the model used for experiment
                you want to continue. Should be of th form 'saved/model_file_name'.

    Returns
    -------
    Saves new .inter, model_top_k.txt model_accepted_songs.txt files to use for further evaluation
    """
    config_dict = {
       'train_neg_sample_args': None,
    }
    config_file_list=['recbole_config.yaml']  # adapt accordingly
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )  # create config
    cont_group = None
    if freeze_country:
        fu = pd.read_csv(dataset_root / dataset / 'demo.txt.gz', sep='\t', header=None,
                         names=['country', 'age', 'gender', 'time'])
        c_freeze = fu.copy().loc[fu['country'] == country]
        c_ind = list(c_freeze.index)
        del fu

    # count = 1  # counter to keep track of iterations
    # model_path = 'saved/MultiVAE-Mar-25-2023_16-42-42.pth' # if we continue from a previous need to define it for model we want to use
    while count <= loops:
        if count == 1:  # if start a new experiment first train the model on the original dataset
            # os.system(f'python dataset_io/create_datasets.py {dataset}')
            print('Running recbole on the new .inter file.')
            run_recbole(model=model, dataset=dataset, config_file_list=['recbole_config.yaml'])  # train the model
            print('Accessing the most recent model file.')
            list_of_models = glob.glob('saved/*.pth')
            model_path = max(list_of_models, key=os.path.getctime)
            print(f'Model used for evaluation : {model_path}')
            model_old = model  # store the model name for later use
            dataset_old = dataset  # store the dataset name for later use
            _, model, dataset, _, _, test_data = load_data_and_model(model_file=model_path)  # get stored model and dataset
        else:  # after the first iteration the model is not retrained only a new dataset is created with new interactions
            # os.system(f'python dataset_io/create_datasets.py {dataset}')
            dataset_old = dataset  # store dataset name for later use
            model_old = model  # store the model name for later
            dataset = create_dataset(config)  # create the new dataset
            train_data, val_data, test_data = data_preparation(config=config, dataset=dataset)  # get datasplits
            _, model, _, _, _, _ = load_data_and_model(model_file=model_path)  # load model from first iteration

        #user_ids, item_ids = _get_ids(dataset)
        #scores_full, actual_full = _get_matrices(user_ids, item_ids, test_data)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        save_preds = True
        k = top_number
        batch_size = 20  # adapt accordingly
        # get scores for model and current dataset
        mean_ndcg = eval_loaded_model(model, dataset, test_data, k, batch_size, device or config['device'], save_preds)

        del model
        del dataset
        del test_data
        model = model_old  # access model name
        print('Scores are being read.')
        #print(dataset)
        dataset = dataset_old  # access dataset name
        scores, actual = read_predictions(dataset_root / dataset / f'{model}.npz')  # get scores
        del actual
        if not rec_repeat:  # if songs are not repeated scores are multiplied with inverse history
            scores = np.nan_to_num(scores, neginf=-1)
            get_history(dataset)
            hist = np.load(dataset_root / dataset / 'history.npy')
            inv_hist = (~hist.astype(bool)).astype(int)
            scores = scores * inv_hist
            del hist
            del inv_hist
        print(f'Top {top_number} items are being calculated.')
        top_k = top_k_item_ids(scores, top_number)  # get top k songs
        top_k.to_csv(dataset_root / dataset / f'{model}_top_k.txt', header=None, sep='\t')  # store recommended songs in txt file
        del scores


        print(f'Top k items saved as {model}_top_k.txt')

        print('Accepted songs are calculated and added to the inter file')
        # load interaction file
        fi = pd.read_csv(dataset_root / dataset / f'{dataset}.inter', sep='\t', header = 0, names=['user_id:token', 'item_id:token'])
        accepted_songs = get_accepted_songs(top_k, conditional, fi.copy(), rec_repeat, top_number, randomly)  # calculate accepted songs and LEs
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
        df_new_recs = pd.concat([fi.copy(), accepted_songs], ignore_index=True)  # add new songs to old interactions
        new_df = df_new_recs.copy().sort_values(['user_id:token', 'item_id:token'])  # sort interactions
        new_df.reset_index(drop=True)
        # new_df = new_df[~new_df.duplicated(subset=['uid','tid'], keep='first')]  # maybe have a look at that again if it would be better to increase Les or something
        del top_k
        del fi
        del df_new_recs
        df = pd.read_csv(dataset_root / dataset / f'{dataset}.train.inter', sep='\t', header=0,
                         names=['user_id:token', 'item_id:token'])
        df_train_recs = pd.concat([df.copy(), accepted_songs], ignore_index=True)
        train_df = df_train_recs.copy().sort_values(['user_id:token', 'item_id:token'])  # sort interactions
        train_df.reset_index(drop=True)
        del df
        del df_train_recs
        del accepted_songs


        if delete:  # delete redundant files if true
            # delete history file
            if os.path.exists(dataset_root / dataset / "history.npy"):
                os.remove(dataset_root / dataset / "history.npy")
            else:
                print(f"The file {dataset}/history.npy does not exist")
            # delete saved scores
            if os.path.exists(dataset_root / dataset / f'{model}.npz'):
                os.remove(dataset_root / dataset / f'{model}.npz')
            else:
                print(f"The file {dataset}/{model}.npz does not exist")
            # delete dataloader
            if os.path.exists(f'saved/{dataset}-for-{model}-dataloader.pth'):
                os.remove(f'saved/{dataset}-for-{model}-dataloader.pth')
            else:
                print(f"The file {dataset}-for-{model}-dataloader.pth does not exist")
            # delete old dataset
            if os.path.exists(f'saved/{dataset}-dataset.pth'):
                os.remove(f'saved/{dataset}-dataset.pth')
            else:
                print(f"The file {dataset}-dataset.pth does not exist")
        # save new interaction file in new folder
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
            model = model_old
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
        config = Config(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
        )  # store new dataset in config
        count += 1  # increase iteration counter


if __name__ == '__main__':
    argh.dispatch_command(loop_without_retraining)
