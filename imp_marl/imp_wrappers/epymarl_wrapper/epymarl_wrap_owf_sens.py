""" Wrapper for owf_sens_env respecting the interface of ePyMARL. """

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    print("")
    print("")
    print("ModuleNotFoundError")
    print("You need to install torch to use the wrapper as in the repository.")
    print("")
    print("")
    raise ModuleNotFoundError

from imp_marl.environments.owf_sens_env import OWF_Sens
from imp_marl.imp_wrappers.pymarl_wrapper.MultiAgentEnv import MultiAgentEnv

class ePymarlOWF_Sens(MultiAgentEnv):
    """
    Wrapper for Struct and Struct_owf respecting the interface of PyMARL.

    It manipulates an imp_env to create all inputs for PyMARL agents.
    """

    def __init__(
        self,
        obs_multiple: bool = False,
        n_owt: int = 2,
        lev: int = 3,
        discount_reward: float = 1.0,
        component_costs: list = [[1, 2, 10], [4, 6, 30]], # [insp, sensor inst, repair]
        global_costs: list = [5, 100, 600], # [mobilization, corrective surplus, system failure]
        mobiliz_elements: int = 5,
        pf_constraint: float = 0.001,
        pf_sys_constraint: float = 1.0,
        sensor_deterioration: list = [[0.02, 0.98, 0.0], [0, 0.35, 0.65], [0.0, 0.0, 1.0]],
        risk_reward: bool = False,
        info_available: str = "all",
        seed=None,
        **kwargs,
    ):
        """
        Initialise based on the full configuration.

        Args:
            obs_multiple: if True, each agent gets the full observation
            n_owt: (int) Number of wind turbines
            lev: (int) Number of levels per wind turbine
            discount_reward: (float) Discount factor [0,1[
            component_costs: (list) Costs associated with each component
            global_costs: (list) Global costs associated with the environment
            mobiliz_elements: (int) Number of mobilization elements
            pf_constraint: (float) Probability of failure constraint
            pf_sys_constraint: (float) Probability of system failure constraint
            sensor_deterioration: (list) Deterioration rates for each sensor
            risk_reward: (bool) If True, use risk-based reward
            info_available: (str) Information available (inspections, monitoring, all)
            seed: (int) Seed for the random number generator
        """
        # Check struct type and default values
        assert (
            isinstance(n_owt, int)
            and n_owt > 0
            and isinstance(discount_reward, float)
            and 0 <= discount_reward <= 1
            and isinstance(obs_multiple, bool)
        ), "Error in env parameters"
        assert 0 <= discount_reward <= 1, "Error in discount_reward"
        assert 0 < mobiliz_elements, "Error in mobiliz_elements"
        assert 0 <= pf_constraint <= 1, "Error in pf_constraint"
        assert 0 <= pf_sys_constraint <= 1, "Error in pf_sys_constraint"

        self.n_owt = n_owt
        self.lev = lev
        self.discount_reward = discount_reward
        self.obs_multiple = obs_multiple
        self.info_available = info_available
        self._seed = seed

        self.config = {
            "n_owt": n_owt,
            "lev": lev,
            "discount_reward": discount_reward,
            "component_costs": component_costs,
            "global_costs": global_costs,
            "mobiliz_elements": mobiliz_elements,
            "pf_constraint": pf_constraint,
            "pf_sys_constraint": pf_sys_constraint,
            "sensor_deterioration": sensor_deterioration,
            "risk_reward": risk_reward,
        }
        self.struct_env = OWF_Sens(self.config)
        self.n_agents = self.struct_env.n_agents

        self.episode_limit = self.struct_env.ep_length
        self.agent_list = self.struct_env.agent_list
        if self.info_available == "inspections":
            self.n_actions = 3  
        elif self.info_available == "monitoring":
            self.n_actions = 4  
        elif self.info_available == "all":
            self.n_actions = self.struct_env.actions_per_agent

        self.action_histogram = {"action_" + str(k): 0 for k in range(self.struct_env.actions_per_agent)}

        self.unit_dim = self.get_unit_dim()  # Qplex requirement

    def update_action_histogram(self, actions):
        """
        Update the action histogram for logging.

        Args:
            actions: list of actions
        """
        for k, action in zip(self.struct_env.agent_list, actions):
            if type(action) is torch.Tensor:
                action_str = str(action.cpu().numpy())
            else:
                action_str = str(action)
            self.action_histogram["action_" + action_str] += 1

    def step(self, actions):
        """
        Ask to run a step in the environment.

        Args:
            actions: list of actions

        Returns:
            rewards: list of rewards
            done: True if the episode is finished
            info: dict of info for logging
        """
        # remapping actions if info_available is limited
        if self.info_available == "inspections":
            if isinstance(actions, list):
                actions = [2 if a == 1 else 4 if a == 2 else a for a in actions]
            elif isinstance(actions, torch.Tensor):
                orig = actions.clone()
                out = actions.clone()
                out[orig == 1] = 2
                out[orig == 2] = 4
                actions = out
            elif isinstance(actions, np.ndarray):
                orig = actions.copy()
                out = actions.copy()
                out[orig == 1] = 2
                out[orig == 2] = 4
                actions = out
            else:
                raise TypeError(f"Unsupported actions type: {type(actions)}")

        elif self.info_available == "monitoring":
            if isinstance(actions, list):
                actions = [4 if a == 2 else 5 if a == 3 else a for a in actions]
            elif isinstance(actions, torch.Tensor):
                orig = actions.clone()
                out = actions.clone()
                out[orig == 2] = 4
                out[orig == 3] = 5
                actions = out
            elif isinstance(actions, np.ndarray):
                orig = actions.copy()
                out = actions.copy()
                out[orig == 2] = 4
                out[orig == 3] = 5
                actions = out
            else:
                raise TypeError(f"Unsupported actions type: {type(actions)}")

        self.update_action_histogram(actions)
        action_dict = {
            k: action for k, action in zip(self.struct_env.agent_list, actions)
        }
        _, rewards, done, _ = self.struct_env.step(action_dict)
        info = {}
        if done:
            for k in self.action_histogram:
                self.action_histogram[k] /= self.episode_limit * self.n_agents
            info = self.action_histogram
        truncated = False
        return self.get_obs(), rewards[self.struct_env.agent_list[0]], done, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_unit_dim(self):
        """Returns the dimension of the unit observation used by QPLEX."""
        return len(self.all_obs_from_struct_env()) // self.n_agents

    def get_obs_agent(self, agent_id: int):
        """
        Returns observation for agent_id

        Args:
            agent_id: id of the agent (int in range(self.n_agents)).
        """
        agent_name = self.struct_env.agent_list[agent_id]

        if self.obs_multiple:
            obs = self.all_obs_from_struct_env()
        else:
            obs = self.struct_env._get_observation()[agent_name]

        return obs

    def get_obs_size(self):
        """Returns the size of the observation."""
        return len(self.get_obs_agent(0))

    def all_obs_from_struct_env(self):
        """Returns all observations concatenated in a single vector."""
        # Concatenate all obs with a single time.
        idx = 0
        obs = None
        for k, v in self.struct_env._get_observation().items():
            if idx == 0:
                obs = v
                idx = 1
            else:
                obs = np.append(obs, v)
        return obs

    def get_state(self):
        """Returns the state of the environment."""
        state = []
        state = np.append(state, self.all_obs_from_struct_env())
        return state

    def get_state_size(self):
        """Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """
        Returns the available actions for agent_id.

        Args:
            agent_id: id of the agent (int in range(self.n_agents)).
        """
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self.action_histogram = {"action_" + str(k): 0 for k in range(self.struct_env.actions_per_agent)}
        self.struct_env.reset()
        return self.get_obs(), self.get_state()

    def render(self):
        """See base class."""
        pass

    def close(self):
        """See base class."""
        pass

    def seed(self):
        """Returns the random seed"""
        return self._seed

    def save_replay(self):
        """See base class."""
        pass

    def get_stats(self):
        """See base class."""
        return {}
