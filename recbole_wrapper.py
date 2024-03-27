import sys

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, construct_transform
from recbole.utils import init_seed, init_logger, get_model, get_flops, set_color, get_trainer, get_environment
from logging import getLogger


def run_recbole_experiment(model: str, dataset: str, iteration: int, config_path: str):
    """
    Initially we used recbole.quick_start.run_recbole() to run the RecBole models.
    However, this has many limitations and undesired behaviour and thus we implemented the function ourselves
    """
    config = Config(model=model, dataset='dataset', config_file_list=[config_path])
    # Recbole hack: set the dataset path to point to the correct subfolder
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
    train_data, valid_data, _ = data_preparation(config, dataset)

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
        train_data, valid_data, saved=False, show_progress=config["show_progress"]
    )
    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")

    return model
