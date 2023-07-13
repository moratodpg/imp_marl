# Wrappers

This package contains wrappers for the IMP environments.

Due to the variety of methods and frameworks used to train agents, we here provide a wrapper for most of them and showcase the flexibility provided by `imp_marl`.

The wrappers simply use the interface defined in [imp_env.py](../imp_env/imp_env.py) in order to plug any IMP environment into any framework.

Examples of the use of these wrappers can be found in the [examples](examples) directories.

## Available wrappers:
- [PymarlMAStruct](pymarl_wrapper/pymarl_wrap_ma_struct.py): Multi-agent wrapper for the [MultiAgentEnv](pymarl_wrapper/MultiAgentEnv.py) interface required by [PyMarl](pymarl_wrapper/README.md).
- [PymarlSAStruct](pymarl_wrapper/pymarl_wrap_sa_struct.py): Single-agent wrapper for the [MultiAgentEnv](SingleAgentEnv.py) interface required by [PyMarl](pymarl_wrapper/README.md).
- [GymSaStruct.py](gymnasium/gym_wrap_sa_struct.py): Single-agent wrapper for the [Gymnasium](https://gymnasium.farama.org/api/env/) interface.
- [PettingZooStruct](pettingzoo/pettingzoo_wrap_struct.py): Multi-agent wrapper for the [PettingZoo](https://pettingzoo.farama.org/) interface.

## Example of wrapper usage:
- [RLLib](examples/rllib/rllib_example.py)