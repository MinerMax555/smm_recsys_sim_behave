import math
import sys
from pathlib import Path
import argh
import bottleneck as bn
import numpy as np
import torch
from argh import arg as cli
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_scores
from tqdm import trange

dataset_root = Path('data')

#from dataset_io.create_datasets import dataset_root


def binary_NDCG(logits: np.ndarray, y_true: np.ndarray, k: int):
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))
    dummy_column = np.arange(len(logits)).reshape((-1, 1))

    idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
    topk_part = logits[dummy_column, idx_topk_part]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[dummy_column, idx_part]

    dcg = (y_true[dummy_column, idx_topk] * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum() for n in np.count_nonzero(y_true, axis=1)])
    return dcg / idcg


def _get_ids(dataset):
    user_ids = list(dataset.field2token_id['user_id'].keys())
    # [PAD] user
    user_ids.remove('[PAD]')
    user_ids = dataset.token2id(dataset.uid_field, user_ids)
    user_ids = user_ids.astype(np.int64)

    item_ids = list(dataset.field2token_id['item_id'].keys())
    # [PAD] item
    item_ids.remove('[PAD]')
    item_ids = dataset.token2id(dataset.iid_field, item_ids)
    item_ids = item_ids.astype(np.int64)

    return user_ids, item_ids


def _get_matrices(user_ids, item_ids, test_data):
    # np.float32 should be high enough precision
    scores_full = np.empty((len(user_ids), len(item_ids)), dtype=np.float32)
    # np.int8 to save memory
    actual_full = np.zeros_like(scores_full, dtype=np.int8)
    # [PAD] user
    for i, items in enumerate(test_data.uid2positive_item[1:]):
        # [PAD] item
        items = items.cpu().numpy() - 1
        actual_full[i, items] = 1

    return scores_full, actual_full


def eval_model(model_path: str, k: int, batch_size: int, device: str, save_preds: bool):
    config, model, dataset, _, _, test_data = load_data_and_model(model_file=model_path)
    return eval_loaded_model(model, dataset, test_data, k, batch_size, device or config['device'], save_preds)


def eval_loaded_model(model, dataset, test_data, k: int, batch_size: int, device: str, save_preds: bool):
    dataset_path = dataset_root / dataset.dataset_name
    model_name = model._get_name()

    model = model.to(device)

    user_ids, item_ids = _get_ids(dataset)
    # explicitly delete to save memory
    del dataset

    scores_full, actual_full = _get_matrices(user_ids, item_ids, test_data)

    ndcgs = []
    for i in trange(math.ceil(len(user_ids) / batch_size), desc=f'Scores & NCDG@10 for {model_name!r}', dynamic_ncols=True, smoothing=0):
        start = i * batch_size
        end = min(len(user_ids), (i + 1) * batch_size)

        batch_scores = full_sort_scores(user_ids[start:end], model, test_data, device=device).cpu().numpy().astype(np.float32)
        scores_full[start:end] = batch_scores[:, item_ids]

        ndcgs.extend(binary_NDCG(
            logits=scores_full[start:end, :],
            y_true=actual_full[start:end, :],
            k=k
        ))

    # Recbole uses its own IDs internally, but before continuing we need to map them back to the original Item IDs
    item_id_mapping = np.zeros(len(item_ids), dtype=np.int64)
    for i in range(0, len(item_ids)):
        # Gets the internal ID within the scores
        recbole_column_index = test_data.dataset.field2token_id['item_id'][str(i)]
        # Recboles Indices start at 1 because they have a [PAD] item at 0
        item_id_mapping[i] = recbole_column_index - 1

    # Get a new view to scores_full with the columns in the correct order
    scores_full = scores_full[:, item_id_mapping]

    if save_preds:
        preds_path = (dataset_path / model_name).with_suffix('.npz')
        print(f'Saving outputs to {preds_path!r} ... (this may take a while)', file=sys.stderr)
        np.savez_compressed(preds_path, scores=scores_full, actual=actual_full)

    return np.mean(ndcgs)


def evaluate_models(models, k=10, batch_size=20, device=None, save_preds=True):
    results = []
    for model in models:
        torch.cuda.empty_cache()
        ndcgs = eval_model(model, k, batch_size, device, save_preds)
        results.append((model, ndcgs.mean()))
    return results

def main():
    # Parse arguments
    parser = argh.ArghParser()
    parser.add_commands([evaluate_models])

    parser.dispatch()

if __name__ == "__main__":
    main()