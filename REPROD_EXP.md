# Reproduce the paper results

This is a step-by-step guide for reproducing of the results presented in the paper.

It may be redundant with the information from others files but the goal here is to have all the instructions in a single file.

# Installation
To reproduce the results in our paper, you will need **python 3.7!!**

Conda:
```
./pymarl/install_conda.sh
```

or Virtual environment (make sure you execute the script with a python 3.7 version):
```
./pymarl/install_venv.sh
```

Several script are in notebook, so you will need to [install jupyter notebook](https://jupyter.org/install).

## Retrieve seeds to execute to reproduce

The seeds required to reproduce our paper are obtained by downloading the logs from our experiments.

To download the logs, you can use the following script or go to the [download instructions](results_scripts/README.md):
```
mkdir -p results_scripts/logs
wget https://zenodo.org/record/8032339/files/MARL_logs.zip
unzip MARL_logs.zip -d results_scripts/logs/
cd logs/MARL_logs
unzip results_scripts/logs/MARL_logs/owf.zip
unzip results_scripts/logs/MARL_logs/struct_c.zip
unzip results_scripts/logs/MARL_logs/struct_uc.zip
```

Once downloaded, you need to find the seeds corresponding to the experiments we made in the paper with the [appropriate notebook](results_scripts/find_seed.ipynb).

The full list of possible algorithm and environment combination can be found [here](EXEC_PYMARL.md).

# Reproduce one experiment

## Train the agents
You want to reproduce the results of QMIX in the k-out-of-n environment with 5 agents and correlations.

Therefore, alg = `qmix_uc_10` and env = `struct_c_5`.

You first need the seeds. Go to the [find seed](results_scripts/find_seed.ipynb), change the `alg` and `env` variable at the last cell and execute the notebook.

You will obtain:

```
480809709
505233690
302946203
545849197
197414500
132396701
608786979
842290689
426355796
850722379
```

To train with the seed=197414500, you need to execute the training script, after activating your environment:

```
conda activate imp_marl_pymarl
python pymarl/train_with_pymarl.py --config=qmix_uc_10 --env-config=struct_c_5 with name=qmix_uc_10_struct_c_5 test_nepisode=-1 seed=197414500
conda deactivate
```

or 

```
source env/bin/activate
python pymarl/train_with_pymarl.py --config=qmix_uc_10 --env-config=struct_c_5 with name=qmix_uc_10_struct_c_5 test_nepisode=-1 seed=197414500
deactivate
````

## Test the agents

A results folder will be created with your results.

You now needs to execute a the tests to obtain the results.

