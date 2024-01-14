# PyMarl

**TO REPRODUCE THE RESULTS IN THE PAPER, PLEASE FOLLOW THE INSTRUCTIONS IN [THIS SECTION](#Reproduce-paper-results).**


The PyMarl code is derived from the [PyMarl original implementation](https://github.com/oxwhirl/pymarl).

# Implemented algorithms

- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**QVMIX**: QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2012.12062)
- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**FACMAC**: Factored Multi-Agent Centralised Policy Gradients](https://arxiv.org/abs/2003.06709)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

# Installation
## Git

Pull the repository:
```
gh repo clone moratodpg/imp_marl
cd imp_marl/papers/neurips_23/pymarl/
```

In this guide, we consider that the current working directory of the terminal is `/imp_marl/papers/neurips_23/pymarl/`.

## Requirements
To reproduce the results in our paper, you will need **python 3.7!!**

Conda:
```
./install_conda.sh
```

Virtual environment:
```
./install_venv.sh
```

## Install the imp_marl packages

You need to install the packages from this repository.

The `-e` allows you to modify imp_marl packages and use them without reinstalling it.

```
pip install -e ../../../ 
```
**Reminder**: the current working directory of the terminal is `/imp_marl/papers/neurips_23/pymarl/`.

# Train and test with pymarl
## Configuration files
Before executing experiments, you need to create the configuration files.
Configuration files are stored in the folder `pymarl/config`.
There are two config folders:
- The [algs](config/algs) folder contains the config for each algorithm.
- The [envs](config/envs) folder contains the config for each environment.

We provide the complete list of configurations used in our paper and you can create new ones on your own.

`config/alg`:

| QMIX       | QVMix        | QPLEX       |   COMA        | FACMAC      | IQL            | DQN      |
|------------|--------------|-------------|---------------|-------------|----------------|----------|
| qmix_uc_10 | qvmix_uc_10  | qplex_uc_10 | coma_uc_10    | facmac_uc_10 | iql_uc_10      | dqn_uc_3 |
| qmix_uc_50 | qvmix_uc_50  | qplex_uc_50 | coma_uc_50    | facmac_uc_50 | iql_uc_50      | dqn_uc_5 |
| qmix_uc_10 | qvmix_uc_100 | qplex_uc_100| coma_uc_100   | facmac_uc_100| iql_uc_100     | /        |

`config/envs`:

| k out of n system | Correlated k out of n system | Off shore wind farm |
|-------------------|------------------------------|---------------------|
| struct_uc_3       | struct_c_3                   | owf_1               |
| struct_uc_5       | struct_c_5                   | owf_2               |
| struct_uc_10      | struct_c_10                  | owf_5               |
| struct_uc_50      | struct_c_50                  | owf_25              |
| struct_uc_100     | struct_c_100                 | owf_50              |

## Train agents

You can train agents, after activating your virtual environment, with the following command:

```
conda activate imp_marl_pymarl
python pymarl/pymarl_train.py --config=alg_config_file --env-config=env_config_file with name=alg_name_in_env_name
conda deactivate
```
Or you can directly use `run.sh` to train agents:

```bash
./run.sh alg_config_name env_config_name
```
Caution: change environment activation lines in `run.sh` to use your own virtual environment.

## Test agents
You can test the agents at training time or after training.

To test the agents at training time, you need to modify `run.sh` file.

To test the agents after training, use the `run_test.sh` file.

```bash
./pymarl_wrapper/run_test.sh experiment_name n_test_epsiode
```

# Reproduce paper results

## Retrieve seeds

The seeds required to reproduce our paper are obtained by downloading the data from our experiments.

These logs are available in [/results_scripts](../results_scripts/).

Once downloaded, the [find_seed notebook](../results_scripts/find_seed.ipynb) allows you to retrieve seeds for a given configuration. 
 
## Train the agents
As example, we want to reproduce the results of QMIX in the k-out-of-n environment with 5 agents and correlations and campaign cost.

Therefore, alg = `qmix_uc_10` and env = `struct_c_5`.

You first need the seeds. Go to the [find seed](../results_scripts/find_seed.ipynb), change the `alg` and `env` variable at the last cell and execute the notebook.

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
python pymarl_train.py --config=qmix_uc_10 --env-config=struct_c_5 with name=qmix_uc_10_struct_c_5 env_args.campaign_cost=True test_nepisode=-1 seed=843209078 
conda deactivate
```
A results folder will be created with your results and you will find the train networks in the `results/models/` folder.

## Test the agents

You now need to execute the test run to get the results, indicating:
- `checkpoint_directory` 
- `tests_number` 
- `campaign_cost_option` 

For this particular example: `qmix_uc_10_struct_c_5__yyyy-mm-dd-hh-mm-ss 10000 True`

```
./run_test.sh qmix_uc_10_struct_c_5__yyyy-mm-dd-hh-mm-ss 10000 True
```
