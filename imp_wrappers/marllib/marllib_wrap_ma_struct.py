# IMP-MARL for MARLLIB
# Coded based on the SMAC implementation of MARLLIB
# https://github.com/Replicable-MARL/MARLlib/blob/master/marllib/envs/base_env/smac.py

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from imp_wrappers.pymarl_wrapper.pymarl_wrap_ma_struct import PymarlMAStruct

import numpy as np
from gym.spaces import Dict as GymDict, Discrete, Box

policy_mapping_dict = {
    "all_scenario": {
        "description": "IMP-MARL all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": False,
    },
}


class MarllibImpMarl(MultiAgentEnv):

    def __init__(self,
                 env_config: dict
                 ):
        struct_type = env_config["struct_type"]
        n_comp = env_config["n_comp"]
        custom_param = env_config["custom_param"]
        discount_reward = env_config["discount_reward"]
        state_obs = env_config["state_obs"]
        state_d_rate = env_config["state_d_rate"]
        state_alphas = env_config["state_alphas"]
        obs_d_rate = env_config["obs_d_rate"]
        obs_multiple = env_config["obs_multiple"]
        obs_all_d_rate = env_config["obs_all_d_rate"]
        obs_alphas = env_config["obs_alphas"]
        env_correlation = env_config["env_correlation"]
        campaign_cost = env_config["campaign_cost"]

        # See in PymarlMAStruct the default values for the parameters
        self.env = PymarlMAStruct(
            struct_type=struct_type,
            n_comp=n_comp,
            custom_param=custom_param,
            discount_reward=discount_reward,
            state_obs=state_obs,
            state_d_rate=state_d_rate,
            state_alphas=state_alphas,
            obs_d_rate=obs_d_rate,
            obs_multiple=obs_multiple,
            obs_all_d_rate=obs_all_d_rate,
            obs_alphas=obs_alphas,
            env_correlation=env_correlation,
            campaign_cost=campaign_cost
        )

        env_info = self.env.get_env_info()
        self.num_agents = self.env.n_agents
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        obs_shape = env_info['obs_shape']
        n_actions = env_info['n_actions']
        state_shape = env_info['state_shape']
        self.observation_space = GymDict({
            "obs": Box(-np.inf, np.inf, shape=(obs_shape,), dtype=np.float64),
            "state": Box(-np.inf, np.inf, shape=(state_shape,),
                         dtype=np.float64),
        })
        self.action_space = Discrete(n_actions)

    def reset(self):
        self.env.reset()
        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()
        obs_dict = {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs_smac[agent_index]
            state_one_agent = state_smac
            agent_index = "agent_{}".format(agent_index)
            obs_dict[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
            }
        return obs_dict

    def step(self, actions):

        actions_ls = [int(actions[agent_id]) for agent_id in actions.keys()]

        reward, terminated, info = self.env.step(actions_ls)

        obs_smac = self.env.get_obs()
        state_smac = self.env.get_state()

        obs_dict = {}
        reward_dict = {}
        for agent_index in range(self.num_agents):
            obs_one_agent = obs_smac[agent_index]
            state_one_agent = state_smac
            agent_index = "agent_{}".format(agent_index)
            obs_dict[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent
            }
            reward_dict[agent_index] = reward

        dones = {"__all__": terminated}

        return obs_dict, reward_dict, dones, {}

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env.episode_limit,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

    def close(self):
        self.env.close()
