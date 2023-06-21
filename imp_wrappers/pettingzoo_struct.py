from imp_wrappers.pettingzoo_env import ParallelEnv

from imp_env.owf_env import Struct_owf
from imp_env.struct_env import Struct

class PettingzooStruct(ParallelEnv):
    metadata = {
        "name": "struct",
    }

    def __init__(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]