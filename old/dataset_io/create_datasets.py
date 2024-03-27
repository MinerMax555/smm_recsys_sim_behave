import gzip
import random
from pathlib import Path

import argh
import numpy as np
import pandas as pd
from argh import arg as cli
from tqdm import tqdm, tqdm


dataset_root = Path('data')


@cli('dataset', type=str, help='dataset name to evaluate')
def main(dataset: str):
    dataset_path = dataset_root / dataset

    inter_file = dataset_path / 'inter.txt.gz'
    history_file = dataset_path / 'history.npy'
    dataset_pandas = pd.read_csv(inter_file, sep='\t', names=['user_id:token', 'item_id:token', 'pc'])

    with gzip.open(inter_file) as in_file:
        data = np.zeros((
                len(pd.unique(dataset_pandas['user_id:token'])),
                len(pd.unique(dataset_pandas['item_id:token']))
            ),
            dtype=np.uint8
        )
        for line in tqdm(in_file, desc=f'reading {inter_file!r}', total=len(dataset_pandas), smoothing=0):
            user, item = map(int, line.decode().split('\t')[:2])
            data[user, item] = 1
    np.save(history_file, data)

    dataset_path.mkdir(exist_ok=True, parents=True)

    random.seed(239476524)
    with open(path := dataset_path / f'{dataset}.inter', 'w') as out_file:
        print('\t'.join(['user_id:token', 'item_id:token']), file=out_file)
        for user_id, row in enumerate(tqdm(data, total=len(data), desc=f'writing {path!r}', smoothing=0)):
            non_zero = list(row.nonzero()[0])
            for i in non_zero:
                print('\t'.join(map(str, [user_id, i])), file=out_file)

if __name__ == '__main__':
    argh.dispatch_command(main)
