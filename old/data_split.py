import sys
import os

import argh
import pandas as pd
import random
from argh import arg as cli
from dataset_io.create_datasets import dataset_root
from recbole.config import Config

random.seed(239476524)


@cli('dataset', type=str, help='dataset name to evaluate')
@cli('-m', '--model', type=str, help='RecBole model used')
def data_splits(dataset: str, model: str = 'ItemKNN'):
    """
    Function to split dataset into train, validation and test set according to recbole_config.yaml in order to have a
    fixed data split for retraining.
    Parameters
    ----------
    dataset: str, Name of the dataset
    model: str, Name of the model

    Returns
    -------
    Saves the splits dataset as dataset.train.inter, dataset.validate.inter, dataset.test.inter
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
    )

    df = pd.read_csv(dataset_root / dataset / f'{dataset}.inter', sep='\t', header = 0, names=['user_id:token', 'item_id:token'])

    # ordering
    ordering_args = config["eval_args"]["order"]
    splits = config["eval_args"]["split"]['RS']
    if splits[0] > 1:
        splits = [x / 100 for x in splits]
    v_t_frac = splits[1] + splits[2]
    v_frac = splits[1]/v_t_frac
    if ordering_args == "RO":
        train = df.groupby('user_id:token').sample(frac=splits[0])
        val_test = pd.concat([df, train]).drop_duplicates(keep=False)
        validate = val_test.groupby('user_id:token').sample(frac=v_frac)
        test = pd.concat([val_test, validate]).drop_duplicates(keep=False)

    train.to_csv(dataset_root / f'{dataset}' / f'{dataset}.train.inter', sep='\t',
                 header=['user_id:token', 'item_id:token'], index=False)
    validate.to_csv(dataset_root / f'{dataset}' / f'{dataset}.validate.inter', sep='\t',
                    header=['user_id:token', 'item_id:token'], index=False)
    test.to_csv(dataset_root / f'{dataset}' / f'{dataset}.test.inter', sep='\t',
                header=['user_id:token', 'item_id:token'], index=False)


if __name__ == '__main__':
    argh.dispatch_command(data_splits)
