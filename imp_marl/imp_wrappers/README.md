# Wrappers for IMP environments

Due to the variety of methods and frameworks used to train agents, we here provide wrappers for most of them and showcase the flexibility provided by `imp_marl`.

The wrappers simply use the interface defined in [imp_env.py](../environments/imp_env.py) in order to plug any IMP environment into any framework.

Examples of the use of these wrappers can be found in [examples](examples).

## Requirements

Each wrapper has its own requirements. Please check the import of each wrapper, where the requirements are listed.

## Available wrappers:
- [PymarlMAStruct](pymarl_wrapper/pymarl_wrap_ma_struct.py): Multi-agent wrapper for the [MultiAgentEnv](pymarl_wrapper/MultiAgentEnv.py) interface required by [PyMarl](pymarl_wrapper/README.md).
- [PymarlSAStruct](pymarl_wrapper/pymarl_wrap_sa_struct.py): Single-agent wrapper for the [MultiAgentEnv](SingleAgentEnv.py) interface required by [PyMarl](pymarl_wrapper/README.md).
- [GymSaStruct](gym/gym_wrap_sa_struct.py): Single-agent wrapper for the [Gym v21 ](https://gymnasium.farama.org/v0.27.1/content/migration-guide/) interface.
- [GymnasiumSaStruct](gymnasium/gymnasium_wrap_sa_struct.py): Single-agent wrapper for the [Gymnasium](https://gymnasium.farama.org/api/env/) interface.
- [PettingZooStruct](pettingzoo/pettingzoo_wrap_struct.py): Multi-agent wrapper for the [PettingZoo](https://pettingzoo.farama.org/) interface.
- [MarllibImpMarl](marllib/marllib_wrap_ma_struct.py): Multi-agent wrapper for [MARLlib](https://github.com/Replicable-MARL/MARLlib).

## Example of wrapper usage:
- [RLLib](examples/rllib/)
- [CleanRL](examples/cleanrl/)
- [MARLlib](examples/marllib/)
