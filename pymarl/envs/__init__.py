from functools import partial
from imp_marl.MultiAgentEnv import MultiAgentEnv
from imp_marl.pymarl_ma_struct import PymarlMAStruct
from imp_marl.pymarl_sa_struct import PymarlSAStruct

import sys
import os



def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["struct_marl"] = partial(env_fn, env=PymarlMAStruct)
REGISTRY["struct_sarl"] = partial(env_fn, env=PymarlSAStruct)
