# Project Overview for Social Media Mining Group

This Readme intends to give a self-written Overview of the Project.

*Written by Max Walder*

## Installation

My basic assumption for this guide is that you are running on Linux.
This makes the installation process much easier.
Make sure you have Python 3.9 or 3.10 installed as well as pip3.

```bash
# Step 1: create a new virtual environment. This makes sure you can install the exact versions of the packages we need in the local folder venv
python3.10 -m venv venv
# Step 2: activate the virtual environment in your current terminal
source venv/bin/activate

# Step 3: Update installation tools to most recent version
python3 -m pip install -U setuptools wheel pip

# Step 4: Install the requirements
# IMPORTANT: You want to install the correct version of torch to have support for your GPU.
# Make sure you look up which CUDA (Nvidia) or ROCM (AMD) version your card supports and choose the correct command:

# CUDA 11.7 (newest version supported by torch 1.13)
python3 -m pip install -U -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
# CUDA 11.6
python3 -m pip install -U -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
# ROCM 5.2 (Linux Support only!)
python3 -m pip install -U -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm5.2
# CPU only
python3 -m pip install -U -r requirements.txt
```

## Quickstart

- always be in the root directory of this project when calling the scripts!
- Make sure your shell has the correct python venv activated (see Step 2 above)
- The following script starts a training loop on the smm_demo dataset (currently just the first of the 6 samplesplits)
  with ItemKNN

```bash
# Quickstart script
# Step 1: activate the virtual environment in your current terminal
source venv/bin/activate

# Step 2: make sure the derivative data folders are removed before starting
rm -r data/smm_demo_*

# Step 3 Perform an example run on the smm_demo dataset.
python3 run_loop.py smm_demo -m ItemKNN -loops 10
```

## Run_loop parameters

| Parameter | Description                                                                                                                               | Default  |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------|----------|
| [dataset] | Subfolder of data/ in which the train, validation and test interactions live. The files must be named i.e. &lt;foldername&gt;.train.inter | N/A      |
| -m        | Model to use. We're interested in "ItemKNN", "BPR", "MultiVAE", "NeuMF" [Full List](https://recbole.io/docs/user_guide/model_intro.html)  | MultiVAE |
| -loops    | Amount of full Train/Eval/Acceptance Loops to generate accepted recommendations to run. Previous Research stopped after 10.               | 20       |
