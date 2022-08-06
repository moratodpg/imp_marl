import itertools

import gym
import numpy as np

from gym import spaces

from struct_env.struct_env import Struct


class GymSaStruct(gym.Env):
    def __init__(self, config=None):
        empty_config = {"config": {"components": 2}}
        config = config or empty_config
        # Number of components #
        self.struct_env = Struct({"components": config['config'].get("components", 2)})
        n_actions = self.struct_env.actions_per_agent
        n_comps = self.struct_env.ncomp
        self.action_space = \
            gym.spaces.Discrete(n_actions ** n_comps)
        self.observation_space = \
            gym.spaces.Box(low=0.0, high=1.0,
                           shape=(self.struct_env.obs_total_single,),
                           dtype=np.float64)

        self.convert_action_dict = {}
        list_actions = \
            list(itertools.product(range(n_actions), repeat=n_comps))
        for idx, i in enumerate(list_actions):
            self.convert_action_dict[idx] = np.array(i)
        print(self.convert_action_dict)
    def reset(self, seed=None, return_info=False, options=None):
        obs_multi = self.struct_env.reset()
        observation = self.convert_obs_multi(obs_multi)
        info = {"belief": self.struct_env.agent_belief}
        return (observation, info) if return_info else observation

    def step(self, action, return_info=False):
        # Here, function is given 1 action from 3 ** n possible actions
        # that need to be converted to 1 action from 3 actions per agent.
        action_multi_list = self.convert_action_dict[action]
        action_multi = {}
        # TODO: maybe remove the dict
        #  from Struct step method to avoid this conversion
        for idx, i in enumerate(self.struct_env.agent_list):
            action_multi[i] = action_multi_list[idx]
        obs_multi, rewards, done = self.struct_env.step(action_multi)
        observation = self.convert_obs_multi(obs_multi)
        reward = rewards[self.struct_env.agent_list[0]]
        info = {"belief": self.struct_env.agent_belief}
        return observation, reward, done, info

    def convert_obs_multi(self, obs_multi):
        time = obs_multi[self.struct_env.agent_list[0]][-1]
        list_obs = [v[:-1] for k, v in obs_multi.items()]
        observation = np.concatenate(list_obs)
        observation = np.append(observation, time)
        return observation
