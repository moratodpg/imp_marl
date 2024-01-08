# Reproduce the paper results

This is a step-by-step guide for reproducing of the results presented in the paper.

It may be redundant with the information from others files but the goal here is to have all the instructions in a single file.

In this guide, we consider that the current working directory of the terminal is the main folder `/imp_marl/`.

# Installation

## Git

Pull the repository:
```
gh repo clone moratodpg/imp_marl
cd imp_marl
```

## Requirements
To reproduce the results in our paper, you will need **python 3.7!!**

Conda:
```
./papers/neurips_23/pymarl/install_conda.sh
```

or Virtual environment (make sure you execute the script with a python 3.7 version):
```
./papers/neurips_23/pymarl/install_venv.sh
```

Several script are in notebook, so you will need to [install jupyter notebook](https://jupyter.org/install).

## Install the packages of imp_marl

You need to install the packages of this repository, **don't forget to activate your python environment**.

The `-e` allows you to modify any package and use it without reinstalling it.

```
pip install -e .
```
# Reproduce one experiment

## Retrieve seeds to reproduce

The seeds required to reproduce our paper are obtained by downloading the logs from our experiments.

To download the logs, you can use the following script or go to the [download instructions](results_scripts/README.md):
```
mkdir -p papers/neurips_23/results_scripts/logs
wget https://zenodo.org/record/8032339/files/MARL_logs.zip
unzip MARL_logs.zip -d results_scripts/logs/
unzip results_scripts/logs/MARL_logs/owf.zip -d results_scripts/logs/MARL_logs/
unzip results_scripts/logs/MARL_logs/struct_c.zip -d results_scripts/logs/MARL_logs/
unzip results_scripts/logs/MARL_logs/struct_uc.zip -d results_scripts/logs/MARL_logs/
```

Once downloaded, you need to find the seeds corresponding to the experiments we made in the paper with the [appropriate notebook](results_scripts/find_seed.ipynb).

The full list of possible algorithm and environment combination can be found [here](pymarl/EXEC_PYMARL.md).

## Train the agents
You want to reproduce the results of QMIX in the k-out-of-n environment with 5 agents and correlations.

Therefore, alg = `qmix_uc_10` and env = `struct_c_5`.

You first need the seeds. Go to the [find seed](results_scripts/find_seed.ipynb), change the `alg` and `env` variable at the last cell and execute the notebook.

You will obtain:

```
604251540
622798568
214664275
846103983
843209078
442054166
512556830
655969730
854845478
540410252
```

To train with the seed=843209078, you need to execute the training script, after activating your environment:

```
conda activate imp_marl_pymarl
python pymarl/pymarl_train.py --config=qmix_uc_10 --env-config=struct_c_5 with name=qmix_uc_10_struct_c_5 test_nepisode=-1 seed=843209078
conda deactivate
```

or 

```
source pymarl/imp_marl_venv/bin/activate
python pymarl/pymarl_train.py --config=qmix_uc_10 --env-config=struct_c_5 with name=qmix_uc_10_struct_c_5 test_nepisode=-1 seed=843209078
deactivate
````

## Test the agents

A results folder will be created with your results and you will find the train networks in the `results/models` folder.

You now needs to execute the test run to obtain the results.

```
./papers/neurips_23/pymarl/run_test.sh qmix_uc_10_struct_c_5__yyyy-mm-dd-hh-mm-ss 10000
```
