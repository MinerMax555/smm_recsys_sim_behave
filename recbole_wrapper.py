import math
import sys

import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, construct_transform
from recbole.utils import init_seed, init_logger, get_model, get_flops, set_color, get_trainer, get_environment
from logging import getLogger

from recbole.utils.case_study import full_sort_scores
from tqdm import trange


def run_recbole_experiment(model: str, dataset: str, iteration: int, config: Config):
    """
    Initially we used recbole.quick_start.run_recbole() to run the RecBole models.
    However, this has many limitations and undesired behaviour and thus we implemented the function ourselves
    """
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # initialize the dataset according to config
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting. Test_data is always empty and thus ignored in our case
    logger.info('Preparing dataset')
    train_data, valid_data, test_data = data_preparation(config, dataset)
    logger.info('Done!')

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model_class = get_model(config["model"])
    # instantiate the model
    model = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")

    # cleanup to hopefully avoid memory leaks
    del model
    del trainer
    del train_data
    del valid_data
    del test_data


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


def get_recbole_scores(model, dataset, test_data, config: Config, batch_size: int = 32):
    user_ids, item_ids = _get_ids(dataset)

    scores = np.empty((len(user_ids), len(item_ids)), dtype=np.float32)

    for i in trange(math.ceil(len(user_ids) / batch_size), desc=f'Calculating Recommendation Scores',
                    dynamic_ncols=True, smoothing=0):
        start = i * batch_size
        end = min(len(user_ids), (i + 1) * batch_size)

        batch_scores = full_sort_scores(user_ids[start:end], model, test_data,
                                        device=config['device']).cpu().numpy().astype(
            np.float32)
        scores[start:end] = batch_scores[:, item_ids]

    # set scores of test set items to -inf such that they are never recommended
    for i, items in enumerate(test_data.uid2positive_item[1:]):
        # -1 because RecBole uses 1-based indexing with a [PAD] item
        items = items.cpu().numpy() - 1
        scores[i, items] = -np.inf

    # Recbole uses its own IDs internally, but before continuing we need to map them back to the original Item IDs
    item_id_mapping = np.zeros(len(item_ids), dtype=np.int64)
    for i in range(0, len(item_ids)):
        # Gets the internal ID within the scores
        recbole_column_index = dataset.field2token_id['item_id'][str(i)]
        # Recboles Indices start at 1 because they have a [PAD] item at 0
        item_id_mapping[i] = recbole_column_index - 1

    # Get a new view to scores_full with the columns in the correct order
    scores = scores[:, item_id_mapping]

    return scores
