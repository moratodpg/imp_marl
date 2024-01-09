import numpy as np
from datetime import datetime
from os import path, makedirs

from imp_marl.environments.struct_env import Struct


class HeuristicsStruct():
    def __init__(self,
                 n_comp: int = 2,
                 # Number of structure
                 discount_reward: float = 1.,
                 # float [0,1] importance of
                 # short-time reward vs long-time reward
                 k_comp: int = None,
                 # Number of structure required (k_comp out of n_comp)
                 env_correlation: bool = False,
                 # env_correlation: True=correlated, False=uncorrelated
                 campaign_cost: bool = False,
                 # campaign_cost = True=campaign cost taken into account
                 seed=None):

        self.n_comp = n_comp
        self.k_comp = k_comp
        self.discount_reward = discount_reward
        self.env_correlation = env_correlation
        self.campaign_cost = campaign_cost
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.config = {"n_comp": n_comp,
                       "discount_reward": discount_reward,
                       "k_comp": k_comp,
                       "env_correlation": env_correlation,
                       "campaign_cost": campaign_cost}
        self.struct_env = Struct(self.config)
        self.date_record = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    def search(self, eval_size):
        insp_interval = np.arange(1, self.struct_env.ep_length)
        comp_inspection = np.arange(1, self.struct_env.n_comp+1)
        heur = np.meshgrid(insp_interval, comp_inspection)
        insp_list = heur[0].reshape(-1)
        comp_list = heur[1].reshape(-1)
        ret_opt = -10000
        ind_opt = 0
        ret_total = []
        for ind in range(len(insp_list)):
            return_heur = 0
            for _ in range(eval_size):
                return_heur += self.episode(insp_list[ind], comp_list[ind])
            return_heur /= eval_size
            ret_total.append(return_heur)
            if return_heur > ret_opt:
                ret_opt = return_heur
                ind_opt = ind
                print('opt', return_heur, 'insp_int', insp_list[ind], 'n_comp', comp_list[ind])
        self.opt_heur = {"opt_reward_mean": ret_opt,
                         "insp_interv": insp_list[ind_opt],
                         "insp_comp": comp_list[ind_opt]}
        
        if not self.env_correlation:
            corr_file = 'uc'
        else:
            corr_file = 'c'
        if not self.campaign_cost:
            camp_file = 'ref'
        else:
            camp_file = 'camp'
        path_results = "heuristics/Results"
        isExist = path.exists(path_results)
        if not isExist:
            makedirs(path_results)
        np.savez('heuristics/Results/heuristics_'+ str(self.n_comp) + '_' + str(self.k_comp) + corr_file + camp_file + '_' + self.date_record,
                  ret_total = ret_total, opt_heur = self.opt_heur, config=self.config, seed_test=self._seed)
        return self.opt_heur

    def eval(self, eval_size, insp_int, comp_insp):
        self.return_heur = 0
        for ep in range(eval_size):
            self.return_heur += self.episode(insp_int, comp_insp)
            disp_cost = self.return_heur/(ep+1)
            if ep%500==0:
                print('Reward:', disp_cost)
        self.return_heur /= eval_size
        return self.return_heur
    
    def episode(self, insp_int, comp_insp):
        rew_total_ = 0
        done_ = False
        insp_obs = 2
        self.struct_env.reset()
        action = {}
        for agent in self.struct_env.agent_list:
            action[agent] = 0
        while not done_:
            action_ = action.copy()
            if (self.struct_env.time_step%insp_int)==0 and self.struct_env.time_step>0:
                pf = self.struct_env.damage_proba[:, -1]
                inspection_index = (-pf).argsort()[:comp_insp]
                for index in inspection_index:
                    action_[self.struct_env.agent_list[index]] = 1
            if np.sum(insp_obs) < self.struct_env.n_comp*2:
                index_repair = np.where(insp_obs==1)[0]
                if len(index_repair) > 0:
                    for index in index_repair:
                        action_[self.struct_env.agent_list[index]] = 2
            [bel_, rew_, done_, insp_obs] = self.struct_env.step(action_)
            rew_total_ += rew_['agent_0']
        return rew_total_
        
