from csv import QUOTE_NONE
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_io import dataset_root


def read_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    inter = pd.read_csv(root / 'inter.tsv.bz2', sep='\t', names=['uid', 'tid', 'LEs'], dtype=int, quoting=QUOTE_NONE)

    demo = pd.read_csv(root / 'demo.tsv.bz2', sep='\t', names=['country', 'age', 'gender', 'creation_time'], dtype={'country': str, 'age': int, 'gender': str, 'creation_time': str}, quoting=QUOTE_NONE)
    # map to datetime
    demo['creation_time'] = pd.to_datetime(demo['creation_time'])

    tracks = pd.read_csv(root / 'tracks.tsv.bz2', sep='\t', names=['artist', 'track', 'country'], dtype=str, quoting=QUOTE_NONE)
    tracks.index = tracks.index.astype(int)

    return inter, demo, tracks


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as npz:
        return npz['scores'], npz['actual']


def read_all(*, dataset: str, model: str) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]]:
    path = dataset_root / dataset
    inter, demo, tracks = read_data(path)
    scores, actual = read_predictions((path / model).with_suffix('.npz'))
    assert demo.shape[0] == scores.shape[0] == actual.shape[0]
    assert tracks.shape[0] == scores.shape[1] == actual.shape[1]

    return (inter, demo, tracks), (scores, actual)
