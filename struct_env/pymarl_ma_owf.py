import numpy as np

from struct_env.MultiAgentEnv import MultiAgentEnv
from struct_env.struct_owf import Struct_owf


class PymarlMAStruct(MultiAgentEnv):
    def __init__(self,
                 n_owt: int = 2,
                 # Number of structure
                 lev: int = 3, 
                 # Number of levels per wind turbine (fix for now)
                 discount_reward: float = 1.,
                 # float [0,1] importance of
                 # short-time reward vs long-time reward
                 state_obs: bool = True,
                 # State contains the concatenation of obs
                 state_d_rate: bool = False,
                 # State contains the concatenation of drate

                 # Obs contains the obs of the agent by default
                 obs_d_rate: bool = False,
                 # Obs contains the drate of the agent
                 obs_multiple: bool = False,
                 # Obs contains the concatenation of all obs
                 obs_all_d_rate: bool = False,
                 # Obs contains the concatenation of all drate
                 campaign_cost: bool = False,
                 # campaign_cost = True=campaign cost taken into account
                 seed=None):

        assert isinstance(state_obs, bool) \
               and isinstance(state_d_rate, bool) \
               and isinstance(obs_d_rate, bool) \
               and isinstance(obs_multiple, bool) \
               and isinstance(obs_all_d_rate, bool) \
               and isinstance(campaign_cost, bool), "Error in env parameters"
        assert 0 <= discount_reward <= 1, "Error in discount_reward"
        assert not (obs_d_rate and obs_all_d_rate), "Error in env parameters"
        assert state_obs or state_d_rate, \
            "Error in env parameters"

        self.n_owt = n_owt
        self.lev = lev
        self.discount_reward = discount_reward
        self.state_obs = state_obs
        self.state_d_rate = state_d_rate
        self.obs_d_rate = obs_d_rate
        self.obs_multiple = obs_multiple
        self.obs_all_drate = obs_all_d_rate
        self.campaign_cost = campaign_cost
        self._seed = seed

        self.config = {"n_owt": n_owt,
                       "lev": lev,
                       "discount_reward": discount_reward,
                       "campaign_cost": campaign_cost}
        self.struct_env = Struct_owf(self.config)
        self.n_agents = self.struct_env.n_agents

        self.episode_limit = self.struct_env.ep_length
        self.agent_list = self.struct_env.agent_list
        self.n_actions = self.struct_env.actions_per_agent

    def step(self, actions):
        """ Returns reward, terminated, info """
        # actions = list
        action_dict = {k: action for k, action in
                       zip(self.struct_env.agent_list, actions)}
        _, rewards, done = self.struct_env.step(action_dict)
        return rewards[self.struct_env.agent_list[0]], done, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return [self.get_obs_agent(i) for i in
                range(self.n_agents)]

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
                obs = np.append(obs, v[:-1])
                # remove the time from the list if not the first element
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

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        return {}
