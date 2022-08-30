import numpy as np

from struct_env.MultiAgentEnv import MultiAgentEnv
from struct_env.struct_env import Struct


class PymarlMAStruct(MultiAgentEnv):

    def __init__(self,
                 components=2,
                 # Number of structure
                 discount_reward=1.,
                 # float [0,1] importance of
                 # short-time reward vs long-time reward
                 k_comp=None,
                 # Number of structure required (k_comp out of components)
                 state_config="obs",
                 # State config ["obs", "drate", "all"]
                 # Caution: obs = all observations from struct_env,
                 # not a concatenation of self.get_obs() !
                 obs_config="single_obs",
                 # Obs config=["so",  # single_obs
                 # "so_sd",  # single_obs + single_drate
                 # "ao",  # all obs
                 # "ao_sd",  # all_obs + single drate
                 # "ao_ad"], # all_obs + all drate
                 seed=None):

        assert obs_config in ["so",
                              "so_sd",
                              "ao",
                              "ao_sd",
                              "ao_ad"], \
            "Error in obs config"
        assert state_config in ["obs", "drate", "all"], \
            "Error in state config"
        assert k_comp is None or k_comp <= components, \
            "Error in k_comp"

        self.discount_reward = discount_reward
        self.state_config = state_config
        self.obs_config = obs_config
        self._seed = seed
        self.config = {"components": components,
                       "discount_reward": discount_reward,
                       "k_comp": k_comp}
        self.struct_env = Struct(self.config)
        self.n_agents = self.struct_env.ncomp
        self.n_comp = self.struct_env.ncomp
        self.k_comp = self.struct_env.k_comp
        self.episode_limit = self.struct_env.ep_length
        self.n_actions = self.struct_env.actions_per_agent
        self.agent_list = self.struct_env.agent_list

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
        if self.obs_config.startswith("so"):
            obs = self.struct_env.observations[agent_name]
            if self.obs_config == "so_sd":
                obs = np.append(obs, self.get_normalized_drate()[agent_id])
            return obs

        elif self.obs_config.startswith("ao"):
            obs = self.all_obs_from_struct_env()

            if self.obs_config == "ao_sd":
                obs = np.append(obs, self.get_normalized_drate()[agent_id])
            elif self.obs_config == "ao_ad":
                obs = np.append(obs, self.get_normalized_drate())
            return obs
        return None

    def get_normalized_drate(self):
        return self.struct_env.drate / self.struct_env.ep_length

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

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.struct_env.obs_per_agent_multi

    def get_state(self):
        if self.state_config == "obs":
            return self.all_obs_from_struct_env()
        elif self.state_config == "drate":
            return self.get_normalized_drate()
        elif self.state_config == "all":
            obs = self.all_obs_from_struct_env()
            drate = self.get_normalized_drate()
            return np.append(obs, drate)
        else:
            print("Error state_config")
            return None

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
