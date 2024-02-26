from functools import partial

from imp_marl.imp_wrappers.pymarl.MultiAgentEnv import MultiAgentEnv
from imp_marl.imp_wrappers.pymarl.pymarl_wrap_ma_struct import PymarlMAStruct
from imp_marl.imp_wrappers.pymarl.pymarl_wrap_sa_struct import PymarlSAStruct


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["struct_marl"] = partial(env_fn, env=PymarlMAStruct)
REGISTRY["struct_sarl"] = partial(env_fn, env=PymarlSAStruct)
