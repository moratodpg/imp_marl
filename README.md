# IMP-MARL: a Suite of Environments for Large-scale Infrastructure Management Planning via MARL
![imp](imp_intro.png)

**Abstract**: *We introduce IMP-MARL, an open-source suite of multi-agent reinforcement learning (MARL) environments for large-scale Infrastructure Management Planning (IMP), offering a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications. In IMP, a multi-component engineering system is subject to a risk of failure due to its components' damage condition. Specifically, each agent plans inspections and repairs for a specific system component, aiming to minimise maintenance costs while cooperating to minimise system failure risk. Through IMP-MARL, we encourage the implementation of new environments and the further development of MARL methods.*

Set of environments available:
- (Correlated and uncorrelated) k-out-of-n system with components subject to fatigue deterioration.
- Offshore wind structural system with components subject to fatigue deterioration.
*A campaign cost can be activated in any environment.

MARL algorithms available:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**QVMIX**: QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2012.12062)
- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**FACMAC**: Factored Multi-Agent Centralised Policy Gradients](https://arxiv.org/abs/2003.06709)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

Expert-knowledge baselines available:
- [Expert-based heuristic strategies](https://www.sciencedirect.com/science/article/pii/S0167473017302138)

Main developers: Pascal Leroy & Pablo G. Morato.

The main code is derived from [pymarl](https://github.com/oxwhirl/pymarl).

## Main requirements:
pymarl:
`python  3.7`
and
`pip install -r requirements.txt` 

## Installation

## Run a simple experiment 

```shell
python3 main.py --config=qmix --env-config=struct with env_args.n_comp=10 env_args.custom_param.k_comp=9
```         
## Tests

## Tutorials
- [Create your own environment scenario](imp_marl/imp_add_env.md)
- [IMP's API explained](imp_wrappers/wrapper_explained.md)
- [Reproduce the reported results](./results_scripts/results_reproduce.md)
- [Directly retrieve the results](./results_scripts/results_retrieve.md)

## Citation
