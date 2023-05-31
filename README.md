# Infrastructure management planning (IMP): An environment for large-scale cooperative MARL methods
![imp](./wrappers/imp_intro.png)

**Abstract**: *In this paper, we introduce an open-source multi-agent reinforcement learning (MARL) environment for Infrastructure Management Planning (IMP), offering a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications. In IMP, each agent should plan inspection and repair actions on a component in order to control the risk of a multi-component engineering system failure, influenced by the components' damage condition. While the cost associated with local maintenance actions should be minimised, all agents must also effectively cooperate to minimise the common system risk. Supported by IMP practical engineering settings featuring up to 100 agents, we conduct an extensive benchmark study, where the scalability and optimality of state-of-the-art cooperative (MA)RL methods are compared against expert knowledge heuristics decision rules. The results reveal that centralised training with decentralised execution methods scale better with the number of agents than fully centralised or decentralised RL approaches, while also outperforming expert knowledge-based management strategies in most IMP settings. Based on our findings, we additionally outline remaining cooperation and scalability challenges that future MARL methods should still address. Through IMP, we encourage both the implementation of additional engineering systems and further development of (MA)RL algorithms in a common publicly available simulation framework.*

Currently available system environment settings:
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

Main developers: Pascal Leroy & Pablo G. Morato.

The main code is derived from [pymarl](https://github.com/oxwhirl/pymarl).

## Main requirements:
pymarl:
`python  3.7`
and
`pip install -r requirements.txt` 

## Installation

## Run an experiment 

```shell
python3 main.py --config=qmix --env-config=struct with env_args.n_comp=10 env_args.custom_param.k_comp=9
```         

## Citation
