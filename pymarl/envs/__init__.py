from functools import partial
from struct_env.MultiAgentEnv import MultiAgentEnv
from struct_env.pymarl_ma_struct import PymarlMAStruct
from struct_env.pymarl_ma_owf import PymarlMAOwf
from struct_env.pymarl_sa_struct import PymarlSAStruct

import sys
import os



def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["struct"] = partial(env_fn, env=PymarlMAStruct)
REGISTRY["owf"] = partial(env_fn, env=PymarlMAOWf)
REGISTRY["struct_sarl"] = partial(env_fn, env=PymarlSAStruct)
