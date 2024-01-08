# Using pymarl wrapper to execute MARL algorithms

**TO REPRODUCE THE RESULTS IN THE PAPER, PLEASE FOLLOW THE INSTRUCTIONS IN [REPROD_EXP.md](../REPROD_EXP.md).**

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
./pymarl/install_conda.sh
```

Virtual environment:
```
./pymarl/install_venv.sh
```

## Install the packages of imp_marl

You need to install the packages of this repository.

The `-e` allows you to modify any package and use it without reinstalling it.

```
pip install -e .
```


# Configuration files
All the configuration files are stored in the folder `pymarl/config`.
There are two config folders: [alg](pymarl/config/alg) and [envs](config/envs).
The `alg` folder contains the config for each algorithm.
The `envs` folder contains the config for each environment.

We provide the complete list of configurations used in our paper and you can create new ones on your own.

# Train agents
Command to train agents, after activating your virtual environment:
```
python pymarl/pymarl_train.py --config=alg_config_file --env-config=env_config_file with name=alg_name_in_env_name
```

# Train agents like in the paper
To train like we did in the paper, we have several config files in the `config/alg` folder.

| QMIX       | QVMix       | QPLEX       |   COMA        | FACMAC      | IQL            | DQN      |
|------------|-------------|-------------|---------------|-------------|----------------|----------|
| qmix_uc_10 | qmix_uc_10  | qplex_uc_10 | coma_uc_10    | facmac_uc_10 | iql_uc_10      | dqn_uc_3 |
| qmix_uc_50 | qmix_uc_50  | qplex_uc_50 | coma_uc_50    | facmac_uc_50 | iql_uc_50      | dqn_uc_5 |
| qmix_uc_10 | qmix_uc_100 | qplex_uc_100| coma_uc_100   | facmac_uc_100| iql_uc_100     | /        |

We also have the corresponding config files in the `config/envs` folder.

| k out of n system | Correlated k out of n system | Off shore wind farm |
|-------------------|------------------------------|---------------------|
| struct_uc_3       | struct_c_3                   | owf_1               |
| struct_uc_5       | struct_c_5                   | owf_2               |
| struct_uc_10      | struct_c_10                  | owf_5               |
| struct_uc_50      | struct_c_50                  | owf_25              |
| struct_uc_100     | struct_c_100                 | owf_50              |

Use the `run.sh` file to train the agents like in the paper:

Caution: Uncomment the right line in `run.sh` to use your own virtual environment.

```bash
./pymarl_wrapper/run.sh alg_config_name env_config_name
```

# Test the agents
You can test the agents at training time or after training.

To test the agents at training time, you need to modify `run.sh` file.

To test the agents at testing time, use the `run_test.sh` file.

```bash
./pymarl_wrapper/run_test.sh experiment_name n_test_epsiode
```

# Reproduce our results
To reproduce the results in the paper, follow the instructions in [REPROD_EXP.md](../REPROD_EXP.md).