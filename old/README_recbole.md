# recbole-for-everyone & Simulation Framework

This repository serves to be a quick-start guide for a Simulation Framework using the RecBole library.
First follow the instructions below to create a virtual environment for using RecBole library. You may need
to do some adjustments to get it working on GPU (the lib is torch-based).

Check 'data/10k_samples_seeds' and 'data/BPR_US_PROB1' for toy data and some idea on what the result should look like.
The results are processed via 'creating_plots_new.py'

 - hint1: if you get weird errors from RecBole - clear the 'saved' dir.
 - hint2: 'run_loop_-_visions_of_future.py' - WIP refactoring with most important functions left and some quality-of-life improvements. Use as a reference or ignore.

---

The basic structure is as follows:

```bash
.
├── data # 'data_path' in recbole_config.yaml
│   └── dataset # 'dataset' in recbole_config.yaml
│       ├── dataset.inter # has to have same name as parent directory
│       ├── dataset.item.txt.gz
│       └── dataset.user.txt.gz
├── dataset_io # helper files for creating and reading custom datasets
│   ├── create_datasets.py
│   ├── __init__.py
│   └── io_helper.py
├── log/... # recbole's log directory
├── log_tensorboard/... # recbole's log directory for tensorboard
├── README.md # the file you're currently viewing
├── recbole_config.yaml # configuration used by recbole
├── recbole_wrappers # helper files for running and evaluating recbole
│   ├── eval_recbole.py # evaluates runs based on NDCG@k
│   ├── __init__.py
│   └── run_recbole.py # runs recbole model
├── requirements.txt # pip-requirements
├── saved/... # recbole's directory to save models and dataloaders
├── loop_helpers.py  # helper files for executing the loop
├── loop_without_retr.py  # file for runnning loop without retraining of the loop
├── run_loop.py  # file for runnning loop
├── data_split.py  # file for creating a fixed data split
├── create_plots_new.ipybn # jupyter notebook to create plots
```
## Using a fixed data split
If we want to use a fixed data split for the experiments we need to add the according files so the data files has the following structure:
```bash
.
├── data # 'data_path' in recbole_config.yaml
│   └── dataset # 'dataset' in recbole_config.yaml
│       ├── dataset.inter # has to have same name as parent directory
│       ├── dataset.train.inter
│       ├── dataset.validate.inter
│       ├── dataset.test.inter
│       ├── dataset.item.txt.gz
│       └── dataset.user.txt.gz
```
We need to define the files in the recbole_config.yaml file accordingly by adding the following line:
```bash
benchmark_filename: ['train', 'validate', 'test']
```
We can create the data split by runnning the data_split.py file
```bash
python data_split.py dataset
```
If we now run the loop the data split stays fixed and we only add new items to the training interactions.
## set up environment

```bash
# create a venv with python3.10
# (recbole does not support python3.11 yet)
# (python3.9 works as well)
# (helper files are written with >=python3.9 syntax features, so <=python3.8 is insufficient)
python3.10 -m venv venv

# activate it
source venv/bin/activate

# update setuptools, wheel and pip itself
python3 -m pip install -U setuptools wheel pip

# load requirements
# depending on the machine you might need to pass '--extra-index-url https://download.pytorch.org/whl/cu<CUDA VERSION>' to get a tourch build with CUDA-Toolkit support; for example:
# python3 -m pip install -U -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
# to get torch with CUDA-11.7 (currently the newest version supported by torch)
python3 -m pip install -U -r requirements.txt

# also note that on windows there are no PyPi-packages available for ray yet
# nightly builds from https://docs.ray.io/en/latest/ray-overview/installation.html#daily-releases-nightlies should be sufficient
```

# Simulation: example run 

```bash
# we always work from the root-directory of the project. All paths are relative to it!

# export PYTHONPATH to current working directory
export PYTHONPATH="."

# activate venv (in case you haven't already)
source venv/bin/activate

# run main file with your dataset
# note that every parameter passed like this will OVERWRITE parameters in 'recbole_config.yaml' !
# Explanations of the parameters can be found using help
python run_loop.py dataset [-m MODEL] [-t LOOPS] [-k TOP_NUMBER]
                   [--freeze_country] [-cc COUNTRY]
                   [--control_group] [-p CONTROL_PERC]
                   [--conditional]
                   [--rec_repeat] [--delete | --no-delete]
                   [--randomly]
                   [-c COUNT]

# The output should be mainly RecBole output with additional info of the current steps

# run the loop_without_retr.py the same way for calculating the baseline approach

# for both we get new files in the dataset

```
## Parameter Description
    dataset: str, Dataset name to evaluate, if you continue previous experiment it needs to be the name of the last
             dataset created
    model: str, Model to use for evaluation
    loops: int, Number of iterations to perform
    top_number: int, number of top items to consider
    freeze_country: bool, If true, songs are not recommended to users coming from the country defined in country.
                    do not so True if control_group is True
    country: str, Country code of country for which users should not get new recommendations, only used when
             freeze_country is True
    control_group: bool, If True, then a randomly selected subgroup of users does not receive recommendations throughout
                   the loop iterations, do not set to True if freeze_country is True
    control_perc: float, between 0 and 1, defines the percentage of users selected for the control_group, only used if
                  control_group is True
    conditional: bool, True if accepted songs should be accepted according to probability, False if all of them
                 should be accepted, do not set to True if randomly is True
    rec_repeat: bool, True if songs should be allowed to be repeated, False if only new songs should be recommended
    delete: bool, True if redundant files should be deleted while running (does not affect original files and new inter
            and top_k.text files), False if all files should be saved while running, Default = True
    randomly: bool, User is modeled by randomly choosing one item from the top k items if True, do not set to True when 
              conditional is True
    count:  Only change when you continue a previously started experiment, needs to be 'count of the last iteration' + 1
            and needs to be smaller than loops


```bash
data
├──dataset
│   ├── dataset.inter
│   ├── dataset.item.txt.gz
│   ├── model_top_k.txt  # file with top k items for each user
│   ├── model_accepted_songs.txt  # file with accepted songs for each user
│   ├── dataset.inter  # interaction file for recbole
│   └── dataset.user.txt.gz
└── dataset_1  # folder of new iteration
    └── dataset.inter  # new interaction history

```

```bash
# investigate predictions with:
python3 -q
```

```py
# load 'data_path'
>>> from dataset_io import dataset_root
# load helper function
>>> from dataset_io.io_helper import read_predictions

# access correct file with dataset_root / <'dataset'> / '<model>.npz'
>>> dataset = 'ml-1m'
>>> model = 'BPR'
>>> scores, actual = read_predictions((dataset_root / dataset / model).with_suffix('.npz'))

# scores as np.float32 matrix
# scores of -inf indicate that this <user, item>-combination was part of the train set
>>> scores.shape
(6040, 3706)

# actual as np.int8 matrix (to save memory)
>>> actual.shape
(6040, 3706)
```
"# recsys-bias-propagation-simulation" 
