from typing import Tuple

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from imp.struct_env import Struct


class RayMaStruct(MultiAgentEnv):

    def __init__(self, config=None):
        empty_config = {"config": {"components": 2}}
        config = config or empty_config
        self.struct_env = Struct({"components": config['config']["components"]})
        self.ncomp = self.struct_env.n_comp

        self.action_space \
            = gym.spaces.Discrete(self.struct_env.actions_per_agent)
        self.observation_space = \
            gym.spaces.Box(low=0.0, high=1.0,
                           shape=(
                               self.struct_env.obs_per_agent_multi,),
                           dtype=np.float64)

    def reset(self) -> MultiAgentDict:
        return self.struct_env.reset()

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        observations, rewards, done, _ = self.struct_env.step(action_dict)
        return observations, rewards, {"__all__": done}, {}
