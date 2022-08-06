from functools import partial
from struct_env.MultiAgentEnv import MultiAgentEnv
from struct_env.pymarl_ma_struct import PymarlMAStruct

import sys
import os




def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["struct"] = partial(env_fn, env=PymarlMAStruct)