import itertools

import numpy as np

from imp_wrappers.pymarl_wrapper.pymarl_wrap_ma_struct import PymarlMAStruct


class PymarlSAStruct(PymarlMAStruct):
    def __init__(self, *args, **kwargs):
        kwargs["obs_d_rate"] = False  # obs are not considered in SARL
        kwargs["obs_multiple"] = False  # obs are not considered in SARL
        kwargs["obs_all_d_rate"] = False  # obs are not considered in SARL
        kwargs["obs_alphas"] = False  # obs are not considered in SARL
        super().__init__(*args, **kwargs)

        n_actions = self.struct_env.actions_per_agent
        self.convert_action_dict = {}
        list_actions = \
            list(itertools.product(range(n_actions), repeat=self.n_agents))
        for idx, i in enumerate(list_actions):
            self.convert_action_dict[idx] = np.array(i)
        self.n_actions = self.struct_env.actions_per_agent = len(list_actions)
        self.action_histogram = {"action_" + str(k): 0 for k in
                                 range(self.n_actions)}
        self.n_agents = 1

    def convert_obs_multi(self, obs_multi):
        time = obs_multi[self.struct_env.agent_list[0]][-1]
        list_obs = [v for k, v in obs_multi.items()]
        observation = np.concatenate(list_obs)
        observation = np.append(observation, time)
        return observation

    def step(self, actions):
        """Returns reward, terminated, info."""
        # actions = a single action
        self.update_action_histogram(actions)
        converted_action = self.convert_action_dict[int(actions[0])]
        action_dict = {k: action for k, action in
                       zip(self.struct_env.agent_list, converted_action)}
        _, rewards, done, _ = self.struct_env.step(action_dict)
        info = {}
        if done:
            for k in self.action_histogram:
                self.action_histogram[k] /= self.episode_limit * self.n_agents
            info = self.action_histogram
        return rewards[self.struct_env.agent_list[0]], done, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.get_state()  # because a single agent!

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.get_state()  # because a single agent!

    def get_obs_size(self):
        """Returns the size of the observation."""
        return len(self.get_obs())

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions
