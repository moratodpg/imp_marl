from struct_env.MultiAgentEnv import MultiAgentEnv
from struct_env.struct_env import Struct


class PymarlMAStruct(MultiAgentEnv):

    def __init__(self,
                 components=2,  # Number of structure
                 discount_reward=1.,
                 k_comp=1,
                 # float [0,1] importance of short-time reward vs long-time reward
                 state_config="obs",  # State config ["obs", "belief"]
                 seed=None):
        self.discount_reward = discount_reward
        self.state_config = state_config
        self._seed = seed
        self.config = {"components": components,
                       "discount_reward": discount_reward,
                       "k_comp": k_comp}
        self.struct_env = Struct(self.config)
        self.n_agents = self.struct_env.ncomp
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
        return [v for k, v in self.struct_env.observations.items()]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.struct_env.observations[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.struct_env.obs_per_agent_multi

    def get_state(self):
        # TODO: state = full obs is not very usefuel in CTDE
        if self.state_config == "obs":
            return [i for j in self.get_obs() for i in j]
        elif self.state_config == "drate":
            return [i / self.struct_env.ep_length for j in
                    self.struct_env.drate for i in j]
        elif self.state_config == "all":
            obs = [i for j in self.get_obs() for i in j]
            drate = [i / self.struct_env.ep_length for j in
                     self.struct_env.drate for i in j]
            return obs + drate
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
