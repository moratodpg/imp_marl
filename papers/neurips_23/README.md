# IMP-MARL: a Suite of Environments for Large-scale Infrastructure Management Planning via MARL

This folder contains the code to reproduce the results of the [paper published at NeurIPS 2023](https://arxiv.org/abs/2306.11551).


- [Train your own MARL agents with PyMarl as in the paper](pymarl/README.md)
- [Reproduce the results of the paper](pymarl/README.md)


## PyMarl algorithms available:

To train agents with PyMarl and one of the following algorithms, instructions are available [here](pymarl/EXEC_PYMARL.md):

- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**QVMIX**: QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2012.12062)
- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**FACMAC**: Factored Multi-Agent Centralised Policy Gradients](https://arxiv.org/abs/2003.06709)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

The main code is derived from [PyMarl original implementation](https://github.com/oxwhirl/pymarl).

## Expert-knowledge baselines available:
- [Expert-based heuristic strategies](https://www.sciencedirect.com/science/article/pii/S0167473017302138)