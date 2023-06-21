# Wrappers

This directory contains wrappers for the IMP environments.

Due to the variety of methods and frameworks used to train agents, we here provide a wrapper for most of them and showcase the flexibility provided by `imp_marl`.

The role of the wrapper is simply to use the interface defined in [imp_env.py](../imp_env/imp_env.py) in order to plug the environment in any framework.

## Available wrappers:
- [PymarlMAStruct.py](pymarl_ma_struct.py): Multi-agent wrapper for the [MultiAgentEnv](MultiAgentEnv.py) interface required by [PyMarl](pymarl/README.md).
- [PymarlSAStruct.py](pymarl_sa_struct.py): Single-agent wrapper for the [MultiAgentEnv](SingleAgentEnv.py) interface required by [PyMarl](pymarl/README.md).
- [gym_sa_struct.py](gym_sa_struct.py): Single-agent wrapper for the [gym](https://gym.openai.com/) interface.