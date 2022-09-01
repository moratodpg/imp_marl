from functools import partial
from struct_env.MultiAgentEnv import MultiAgentEnv
from struct_env.pymarl_ma_struct import PymarlMAStruct
from struct_env.pymarl_sa_struct import PymarlSAStruct
from struct_env.pymarl_ma_struct_corr import PymarlMAStruct_corr

import sys
import os



def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["struct"] = partial(env_fn, env=PymarlMAStruct)
REGISTRY["struct_sarl"] = partial(env_fn, env=PymarlSAStruct)
REGISTRY["struct_corr"] = partial(env_fn, env=PymarlMAStruct_corr)