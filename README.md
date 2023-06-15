# IMP-MARL: a Suite of Environments for Large-scale Infrastructure Management Planning via MARL
![imp](imp_intro.png)

**IMP-MARL** offers a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications.

In IMP-MARL, you can:
- Implement your own infrastructure management planning (IMP) environment or execute an IMP environment available => [imp_env](./imp_env/)
- Generate IMP policies through state-of-the-art MARL methods. The environments can be integrated with typical ecosystems via wrappers => [imp_wrappers](./imp_wrappers/)
- Compute expert-based heuristic policies => [heuristics](./heuristics/)
Additionally, you can also:
- Retrieve the results of a benchmark campaign, where MARL methods are assessed in terms of scalibility.
- Reproduce our experiments.

## Main requirements:
pymarl:
`python  3.7`
and
`pip install -r requirements.txt` 

## Installation:


## Sets of environments available:
- (Correlated and uncorrelated) k-out-of-n system with components subject to fatigue deterioration => [struct](./imp_env/struct_env.py)
- Offshore wind structural system with components subject to fatigue deterioration. => [owf](./imp_env/owf_env.py)

*A campaign cost can be activated in any environment.

## MARL algorithms available:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**QVMIX**: QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2012.12062)
- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**FACMAC**: Factored Multi-Agent Centralised Policy Gradients](https://arxiv.org/abs/2003.06709)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

The main code is derived from [pymarl](https://github.com/oxwhirl/pymarl).

## Expert-knowledge baselines available:
- [Expert-based heuristic strategies](https://www.sciencedirect.com/science/article/pii/S0167473017302138)

## Tutorials
- [Create your own environment scenario](./imp_env/new_imp_env_tutorial.ipynb)
- [IMP's API explained](imp_wrappers/wrapper_explained.md)
- [Reproduce the reported results](./results_scripts/README.md)
- [Retrieve directly the results](./results_scripts/README.md)

## Run a simple experiment 

```shell
python3 main.py --config=qmix --env-config=struct with env_args.n_comp=10 env_args.custom_param.k_comp=9
```         

## Citation
Main developers: Pascal Leroy & Pablo G. Morato.