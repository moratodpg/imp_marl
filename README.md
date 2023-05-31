# Infrastructure management planning (IMP): An environment for large-scale cooperative MARL methods
![imp](./wrappers/imp_intro.png)

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

## Run an experiment 

```shell
python3 main.py --config=qmix --env-config=struct with env_args.n_comp=10 env_args.custom_param.k_comp=9
```                                                                                                                                                        
