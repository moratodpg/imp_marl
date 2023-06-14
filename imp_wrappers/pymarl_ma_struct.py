import numpy as np
import torch

from imp_env.owf_env import Struct_owf
from imp_env.struct_env import Struct
from imp_wrappers.MultiAgentEnv import MultiAgentEnv


class PymarlMAStruct(MultiAgentEnv):
    def __init__(self,
                 struct_type: str = "struct",
                 # Type of the struct env, either "struct" or "owf".
                 n_comp: int = 2,
                 # Number of structure
                 custom_param: dict = None,
                 # struct: Number of structure required
                 #      {"k_comp": int} for k_comp out of n_comp
                 #      Default is None, meaning k_comp=n_comp-1
                 # owf: Number of levels per wind turbine
                 #      {"lev": int}
                 #      Default is 3
                 discount_reward: float = 1.,
                 # float [0,1] importance of
                 # short-time reward vs long-time reward

                 state_obs: bool = True,
                 # State contains the concatenation of obs
                 state_d_rate: bool = False,
                 # State contains the concatenation of drate
                 state_alphas: bool = False,
                 # State contains the concatenation of alpha

                 # Obs contains the obs of the agent by default
                 obs_d_rate: bool = False,
                 # Obs contains the drate of the agent
                 obs_multiple: bool = False,
                 # Obs contains the concatenation of all obs
                 obs_all_d_rate: bool = False,
                 # Obs contains the concatenation of all drate
                 obs_alphas: bool = False,
                 # Obs contains the alphas
                 env_correlation: bool = False,
                 # env_correlation: True=correlated, False=uncorrelated
                 campaign_cost: bool = False,
                 # campaign_cost = True=campaign cost taken into account
                 seed=None):

        # Check struct type and default values
        assert struct_type == "owf" or struct_type == "struct", "Error in struct_type"
        if struct_type == "struct":
            self.k_comp = custom_param.get("k_comp", None) if (custom_param is not None) else None
            assert self.k_comp is None or self.k_comp <= n_comp, "Error in k_comp"
        elif struct_type == "owf":
            self.lev = custom_param.get("lev", 3) if (custom_param is not None) else 3
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
        self._seed = seed

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

        self.unit_dim = self.get_unit_dim() # Qplex requirement

    def update_action_histogram(self, actions):
        for k, action in zip(self.struct_env.agent_list, actions):
            if type(action) is torch.Tensor:
                action_str = str(action.cpu().numpy())
            else:
                action_str = str(action)
            self.action_histogram["action_" + action_str] += 1

    def step(self, actions):
        """ Returns reward, terminated, info """
        # actions = list
        self.update_action_histogram(actions)
        action_dict = {k:action
                       for k, action in zip(self.struct_env.agent_list, actions)}
        _, rewards, done, _ = self.struct_env.step(action_dict)
        info = {}
        if done:
            for k in self.action_histogram:
                self.action_histogram[k] /= self.episode_limit * self.n_agents
            info = self.action_histogram
        return rewards[self.struct_env.agent_list[0]], done, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        return [self.get_obs_agent(i) for i in
                range(self.n_agents)]

    def get_unit_dim(self):
        return len(self.all_obs_from_struct_env())//self.n_agents

    def get_obs_agent(self, agent_id):
        """
        Returns observation for agent_id
        agent_id = integer in range (self.n_agents)
        """
        agent_name = self.struct_env.agent_list[agent_id]

        if self.obs_multiple:
            obs = self.all_obs_from_struct_env()
        else:
            obs = self.struct_env.observations[agent_name]

        if self.obs_d_rate:
            obs = np.append(obs, self.get_normalized_drate()[agent_id])

        if self.obs_all_drate:
            obs = np.append(obs, self.get_normalized_drate())

        if self.obs_alphas:
            obs = np.append(obs, self.struct_env.alphas)

        return obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_normalized_drate(self):
        return self.struct_env.d_rate / self.struct_env.ep_length

    def all_obs_from_struct_env(self):
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

    def get_state(self):
        state = []
        if self.state_obs:
            state = np.append(state, self.all_obs_from_struct_env())
        if self.state_d_rate:
            state = np.append(state, self.get_normalized_drate())
        if self.state_alphas:
            state = np.append(state, self.struct_env.alphas)
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1] * self.n_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.struct_env.actions_per_agent

    def reset(self):
        """ Returns initial observations and states"""
        self.action_histogram = {"action_"+str(k): 0 for k in range(self.n_actions)}
        self.struct_env.reset()
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
