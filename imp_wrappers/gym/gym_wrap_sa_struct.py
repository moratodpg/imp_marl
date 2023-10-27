""" Gym wrapper for IMP environments. """

import itertools
import gym
# Example coded with gym==0.23
import numpy as np
from gym import spaces

from imp_env.owf_env import Struct_owf
from imp_env.struct_env import Struct


class GymSaStruct(gym.Env):
    def __init__(self,
                 struct_type: str = "struct",
                 n_comp: int = 2,
                 custom_param: dict = None,
                 discount_reward: float = 1.,
                 state_obs: bool = True,
                 state_d_rate: bool = False,
                 state_alphas: bool = False,
                 obs_d_rate: bool = False,
                 obs_multiple: bool = False,
                 obs_all_d_rate: bool = False,
                 obs_alphas: bool = False,
                 env_correlation: bool = False,
                 campaign_cost: bool = False):
        """
        Initialize the environment like in PyMARL.
        """

        # SARL so disable obs
        obs_d_rate = False
        obs_multiple = False
        obs_all_d_rate = False
        obs_alphas = False

        # Check struct type and default values
        assert struct_type == "owf" or struct_type == "struct", "Error in struct_type"
        if struct_type == "struct":
            self.k_comp = custom_param.get("k_comp", None) if (
                    custom_param is not None) else None
            assert self.k_comp is None or self.k_comp <= n_comp, "Error in k_comp"
        elif struct_type == "owf":
            self.lev = custom_param.get("lev", 3) if (
                    custom_param is not None) else 3
            assert self.lev is not None, "Error in lev"
            obs_alphas = False
            env_correlation = False
            state_alphas = False

        assert isinstance(state_obs, bool) \
               and isinstance(state_d_rate, bool) \
               and isinstance(state_alphas, bool) \
               and isinstance(obs_d_rate, bool) \
               and isinstance(obs_multiple, bool) \
               and isinstance(obs_all_d_rate, bool) \
               and isinstance(obs_alphas, bool) \
               and isinstance(env_correlation, bool) \
               and isinstance(campaign_cost, bool), "Error in env parameters"
        assert 0 <= discount_reward <= 1, "Error in discount_reward"
        assert not (obs_d_rate and obs_all_d_rate), "Error in env parameters"
        assert state_obs or state_d_rate or state_alphas, \
            "Error in env parameters"
        if not env_correlation:
            assert not obs_alphas, \
                "Error in env parameter obs_alphas"
            assert not state_alphas, \
                "Error in env parameter state_alphas"

        self.n_comp = n_comp
        self.custom_param = custom_param
        self.discount_reward = discount_reward
        self.state_obs = state_obs
        self.state_d_rate = state_d_rate
        self.state_alphas = state_alphas
        self.obs_d_rate = obs_d_rate
        self.obs_multiple = obs_multiple
        self.obs_all_drate = obs_all_d_rate
        self.obs_alphas = obs_alphas
        self.env_correlation = env_correlation
        self.campaign_cost = campaign_cost

        if struct_type == "struct":
            self.config = {"n_comp": n_comp,
                           "discount_reward": discount_reward,
                           "k_comp": self.k_comp,
                           "env_correlation": env_correlation,
                           "campaign_cost": campaign_cost}
            self.struct_env = Struct(self.config)
            self.n_agents = self.struct_env.n_comp
        elif struct_type == "owf":
            self.config = {"n_owt": n_comp,
                           "lev": self.lev,
                           "discount_reward": discount_reward,
                           "campaign_cost": campaign_cost}

            self.struct_env = Struct_owf(self.config)
            self.n_agents = self.struct_env.n_agents

        self.episode_limit = self.struct_env.ep_length
        self.agent_list = self.struct_env.agent_list
        self.n_actions = self.struct_env.actions_per_agent

        self.action_histogram = {"action_" + str(k): 0 for k in
                                 range(self.n_actions)}

        n_actions = self.struct_env.actions_per_agent
        self.convert_action_dict = {}
        list_actions = \
            list(itertools.product(range(n_actions), repeat=self.n_agents))
        for idx, i in enumerate(list_actions):
            self.convert_action_dict[idx] = np.array(i)
        self.n_actions = len(list_actions)
        self.n_agents = 1

        self.struct_env.reset()

        # Gymnasium attributes
        self.action_space = spaces.Discrete(self.n_actions)

        state = self.get_state()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(state),),
                                            dtype=np.float32)
        self.render_mode = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.struct_env.reset()
        return self.get_state()

    def step(self, action):
        converted_action = self.convert_action_dict[action]
        action_dict = {k: action for k, action in
                       zip(self.struct_env.agent_list, converted_action)}
        _, rewards, done, _ = self.struct_env.step(action_dict)
        info = {}
        state = self.get_state()
        reward = rewards[self.struct_env.agent_list[0]]
        return state, reward, done, info

    def get_state(self):
        """ Returns the state of the environment. """
        state = []
        if self.state_obs:
            state = np.append(state, self.all_obs_from_struct_env())
        if self.state_d_rate:
            state = np.append(state, self.get_normalized_drate())
        if self.state_alphas:
            state = np.append(state, self.struct_env.alphas)
        return state

    def all_obs_from_struct_env(self):
        """ Returns all observations concatenated in a single vector. """
        # Concatenate all obs with a single time.
        idx = 0
        obs = None
        for k, v in self.struct_env.observations.items():
            if idx == 0:
                obs = v
                idx = 1
            else:
                obs = np.append(obs, v)
        return obs

    def get_normalized_drate(self):
        """ Returns the normalized d_rate. """
        return self.struct_env.d_rate / self.struct_env.ep_length
