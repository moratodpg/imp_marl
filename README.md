# IMP-MARL: a Suite of Environments for Large-scale Infrastructure Management Planning via MARL


**IMP-MARL** offers a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications.

In IMP-MARL, you can:
- [Implement your own infrastructure management planning (IMP) environment or execute an available IMP environment](imp_marl/environments/).
- [Train IMP policies through state-of-the-art MARL methods. The environments can be integrated with typical ecosystems via wrappers](imp_marl/imp_wrappers/).
- [Compute expert-based heuristic policies](papers/neurips_23/heuristics/)

Additionally, you will be able to:
- Retrieve the results of a benchmark campaign, where MARL methods are assessed in terms of scalability.
- Reproduce our experiments.
- Add your results to ours through the plot scripts.

This repository has been developed and is maintained by Pascal Leroy & Pablo G. Morato.

Please consider opening an issue or a pull request to help us improve this repository.

Future developments are described in the [roadmap](ROADMAP.md).

![imp](imp_intro.png)

## Requirements
To work with our environments, one only needs to install [Numpy](https://numpy.org/install/).

However, to reproduce our results, more packages are required and installation instructions are provided [here](papers/neurips_23/pymarl/README.md).

## Tutorials
- [Create your own IMP environment scenario](imp_marl/environments/new_imp_env_tutorial.ipynb)
- [IMP's interface explained](imp_marl/environments/README.md)
- [Train agents like in the paper and/or **reproduce** the results](papers/neurips_23/README.md)
- [Retrieve the results of the paper and execute the plot scripts](papers/neurips_23/results_scripts/README.md)

## Sets of environments available
- [(Correlated and uncorrelated) k-out-of-n system with components subject to fatigue deterioration.](imp_marl/environments/struct_env.py)
- [Offshore wind structural system with components subject to fatigue deterioration.](imp_marl/environments/owf_env.py)

**Note: A campaign cost can be activated in any environment.**

## Available wrappers and examples
All wrappers are available in [imp_wrappers](imp_marl/imp_wrappers/).
- Ready: [**PyMarl**](imp_marl/imp_wrappers/pymarl): [Multi](imp_marl/imp_wrappers/pymarl/pymarl_wrap_ma_struct.py) and [single](imp_marl/imp_wrappers/pymarl/pymarl_wrap_sa_struct.py) agent wrappers.
- Ready: [**EPyMarl**](imp_marl/imp_wrappers/epymarl/epymarl_wrap_ma_struct.py): Multi-agent wrapper.
- Ready: [**Gym**](imp_marl/imp_wrappers/gym/gym_wrap_sa_struct.py): Single-agent wrapper.
- Ready: [**Gymnasium**](imp_marl/imp_wrappers/gymnasium/gymnasium_wrap_sa_struct.py): Single-agent wrapper.
- Ready: [**PettingZoo**](imp_marl/imp_wrappers/pettingzoo/pettingzoo_wrap_struct.py) : Multi-agent wrapper.
- Ready: [**Rllib**](imp_marl/imp_wrappers/examples/rllib/rllib_example.py): Single-agent training with RLLib and Gymnasium wrapper.
- Ready: [**MARLlib**](imp_marl/imp_wrappers/marllib/marllib_wrap_ma_struct.py): Examples include random agents and how to train with MARLlib.
- Ready: [**CleanRL**](imp_marl/imp_wrappers/examples/CleanRL): Examples on how to train with CleanRL using the Gym wrapper.
- WIP: [**TorchRL example**](): WIP


## Run an IMP environment 
```
env = Struct({'n_comp': 3,
               'discount_reward': 0.95,
               'k_comp': 2,
               'env_correlation': False,
               'campaign_cost': False})

obs, done = env.reset(), False
while not done:
    actions = {f"agent_{i}": random.randint(0,2) for i in range(3)}
    obs, rewards, done, insp_outcomes = env.step(actions) 
```   

## Citation
If you use IMP-MARL in your work, please consider citing our paper:

[IMP-MARL: a Suite of Environments for Large-scale Infrastructure Management Planning via MARL](https://arxiv.org/abs/2306.11551)
```
@inproceedings{
leroy2023impmarl,
title={{IMP}-{MARL}: a Suite of Environments for Large-scale Infrastructure Management Planning via {MARL}},
author={Pascal Leroy and Pablo G. Morato and Jonathan Pisane and Athanasios Kolios and Damien Ernst},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2023},
url={https://openreview.net/forum?id=q3FJk2Nvkk}
}
```
