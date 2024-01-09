# Script to execute the marllib wrapper with random actions
import os

import numpy as np
import yaml
from yaml import SafeLoader

from imp_marl.imp_wrappers.marllib.marllib_wrap_ma_struct import MarllibImpMarl

if __name__ == '__main__':

    n_episode = 100

    env_config = {
        "struct_type": "struct",
        "n_comp": 2,
        "discount_reward": .95,
        "custom_param": None,
        "state_obs": True,
        "state_d_rate": False,
        "state_alphas": False,
        "obs_d_rate": False,
        "obs_multiple": False,
        "obs_all_d_rate": False,
        "obs_alphas": False,
        "env_correlation": False,
        "campaign_cost": False}

    env = MarllibImpMarl(env_config)
    array_reward = []
    for i in range(n_episode):
        obs_dict = env.reset()
        terminated = False
        episode_reward = 0
        while not terminated:
            actions = {}
            for k in obs_dict.keys():
                actions[k] = env.action_space.sample()
            obs_dict, reward_dict, dones, _ = env.step(actions)
            terminated = dones["__all__"]
            episode_reward += reward_dict["agent_0"]
        array_reward.append(episode_reward)
    print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
          np.min(array_reward), np.max(array_reward))

    env.close()
