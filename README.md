# Infrastructure management planning (IMP): An environment for large-scale cooperative MARL methods
![imp](./wrappers/imp_intro.png)

**Abstract**: *We introduce an open-source multi-agent reinforcement learning (MARL) environment for Infrastructure Management Planning (IMP), offering a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications. In IMP, each agent should plan inspection and repair actions on a component in order to control the risk of a multi-component engineering system failure, influenced by the components' damage condition. While the cost associated with local maintenance actions should be minimised, all agents must also effectively cooperate to minimise the common system risk. Through IMP, we encourage both the implementation of additional engineering systems and further development of (MA)RL algorithms in a common publicly available simulation framework.*

System environment settings available:
- k-out-of-n system with components subject to fatigue deterioration.
- offshore wind structural system.

MARL algorithms available:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**QVMIX**: QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2012.12062)
- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**FACMAC**: Factored Multi-Agent Centralised Policy Gradients](https://arxiv.org/abs/2003.06709)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

Expert-knowledge baselines available:
- [Heuristic decision rules](https://www.sciencedirect.com/science/article/pii/S0167473017302138)

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
- [Create your own environment scenario](imp_env/imp_add_env.md)
- [IMP's API explained](./wrappers/api_explained.md)
- [Reproduce the reported results](./results_scripts/results_reproduce.md)
- [Directly retrieve the results](./results_scripts/results_retrieve.md)

## Citation
