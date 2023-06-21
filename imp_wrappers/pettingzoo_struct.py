from imp_wrappers.pettingzoo_env import ParallelEnv

from imp_env.owf_env import Struct_owf
from imp_env.struct_env import Struct

class PettingzooStruct(ParallelEnv):
    metadata = {
        "name": "struct",
    }

    def __init__(self,
        struct_type: str = "struct",
        n_comp: int = 2,
        k_comp: int = 1,
        discount_reward: float = 1.,
        env_correlation: bool = False,
        campaign_cost: bool = False,
        seed=None):

        self.n_comp = n_comp
        self.k_comp = k_comp
        self.discount_reward = discount_reward
        self.env_correlation = env_correlation
        self.campaign_cost = campaign_cost
        self._seed = seed


        self.config = {"n_comp": n_comp,
                        "discount_reward": discount_reward,
                        "k_comp": k_comp,
                        "env_correlation": env_correlation,
                        "campaign_cost": campaign_cost}
        
        self.struct_env = Struct(self.config)
        self.n_agents = self.struct_env.n_comp

        self.episode_limit = self.struct_env.ep_length
        self.agent_list = self.struct_env.agent_list
        self.n_actions = self.struct_env.actions_per_agent

        self.action_histogram = {"action_" + str(k): 0 for k in
                                 range(self.n_actions)}

    def reset(self, seed=None, options=None):

        return observations, {}

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]