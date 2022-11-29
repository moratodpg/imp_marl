# Multi-agent (deep) reinforcement learning for structural systems (marl_struct)

Marl_struct offers a framework for DRL algorithms and structural systems environments. In particular, the following algorithms are available:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**QVMIX**: QVMix and QVMix-Max: Extending the Deep Quality-Value Family of Algorithms to Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2012.12062)
- [**DDMAC**: DDMAC: Managing engineering systems with large state and action spaces through deep reinforcement learning](https://arxiv.org/abs/1811.02052)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

Main developers: Pablo G. Morato & Pascal Leroy.

The main code is derived from [pymarl](https://github.com/oxwhirl/pymarl).

## Main requirements:
pymarl:
`python  3.7`
and
`pip install -r requirements.txt` 

## Run an experiment 

```shell
python3 main.py --config=qmix --env-config=struct with env_args.n_comp=10 env_args.custom_param=9
```
