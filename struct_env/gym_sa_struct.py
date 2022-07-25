import itertools

import gym
import numpy as np

from gym import spaces

from struct_env.struct_env import Struct


class GymSaStruct(gym.Env):
    def __init__(self, config=None):
        self.struct_env = Struct(config)
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

    def reset(self, seed=None, return_info=False, options=None):
        obs_multi = self.struct_env.reset()
        observation = self.convert_obs_multi(obs_multi)
        info = {"belief": self.struct_env.agent_belief}
        return (observation, info) if return_info else observation

    def step(self, action, return_info=False):
        # Here, function is given 1 action from 3 ** n possible actions
        # that need to be converted to 1 action from 3 actions per agent.
        action = self.convert_action_dict[action]
        print(action)
        observations, rewards, dones = self.struct_env.step(action)
        print(observations)
        print(rewards)
        print(dones)

    def convert_obs_multi(self, obs_multi):
        time = obs_multi[self.struct_env.agent_list[0]][-1]
        list_obs = [v[:-1] for k, v in obs_multi.items()]
        observation = np.concatenate(list_obs)
        observation = np.append(observation, time)
        return observation

    def convert_base_action(self, action, base, comp):

        # TODO: Hard code an array to have instantaneous answer.
        action_multi = np.zeros((comp,), dtype=int)
        if action == 0:
            return action_multi
        digits = []
        index_comp = int(comp) - 1
        while action:
            digits = (int(action % base))
            action_multi[index_comp] = digits
            action //= base
            index_comp -= 1
        return action_multi
