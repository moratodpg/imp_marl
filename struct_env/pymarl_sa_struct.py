import itertools

import numpy as np

from struct_env.MultiAgentEnv import MultiAgentEnv
from struct_env.struct_env import Struct


class PymarlSAStruct(MultiAgentEnv):
    def __init__(self,
                 components=2,  # Number of structure
                 discount_reward=1.,
                 # float [0,1] importance of short-time reward vs long-time reward
                 state_config="obs",  # State config ["obs", "belief"]
                 seed=None):
        self.discount_reward = discount_reward
        self.state_config = state_config
        self._seed = seed
        self.config = {"components": components,
                       "discount_reward": discount_reward}
        self.struct_env = Struct(self.config)
        self.n_agents = 1
        self.n_comp = self.struct_env.ncomp
        self.episode_limit = self.struct_env.ep_length

        self.agent_list = self.struct_env.agent_list

        n_actions = self.struct_env.actions_per_agent
        self.convert_action_dict = {}
        list_actions = \
            list(itertools.product(range(n_actions), repeat=self.n_comp))
        for idx, i in enumerate(list_actions):
            self.convert_action_dict[idx] = np.array(i)
        self.n_actions = self.struct_env.actions_per_agent=len(list_actions)

    def step(self, actions):
        """Returns reward, terminated, info."""
        # actions = a single action
        converted_action = self.convert_action_dict[int(actions[0])]
        action_dict = {k: action for k, action in
                       zip(self.struct_env.agent_list, converted_action)}
        _, rewards, done = self.struct_env.step(action_dict)
        return rewards[self.struct_env.agent_list[0]], done, {}

    def convert_obs_multi(self, obs_multi):
        time = obs_multi[self.struct_env.agent_list[0]][-1]
        list_obs = [v[:-1] for k, v in obs_multi.items()]
        observation = np.concatenate(list_obs)
        observation = np.append(observation, time)
        return observation

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.convert_obs_multi(self.struct_env.observations)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.get_obs() # because a single agent!

    def get_obs_size(self):
        """Returns the size of the observation."""
        return len(self.get_obs())

    def get_state(self):
        """Returns the global state."""
        if self.state_config == "obs":
            return self.get_obs()
        elif self.state_config == "drate":
            return [i / self.struct_env.ep_length for j in
                    self.struct_env.drate for i in j]
        elif self.state_config == "all":
            obs = self.get_obs()
            drate = np.array([i / self.struct_env.ep_length for j in
                     self.struct_env.drate for i in j])
            return np.concatenate([obs, drate])
        else:
            print("Error state_config")
            return None

    def get_state_size(self):
        """Returns the size of the global state."""
        return len(self.get_state())

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self.struct_env.reset()
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        return self._seed

    def save_replay(self):
        """Save a replay."""
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
