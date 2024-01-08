import numpy as np
from datetime import datetime
from os import path, makedirs

from environments.owf_env import Struct_owf


class HeuristicsOwf():
    def __init__(self,
                 n_owt: int = 2,
                 # Number of structure
                 lev: int = 3,
                 discount_reward: float = 1.,
                 # float [0,1] importance of
                 # short-time reward vs long-time reward
                 campaign_cost: bool = False,
                 # campaign_cost = True=campaign cost taken into account
                 seed=None):

        self.n_owt = n_owt
        self.lev = lev
        self.discount_reward = discount_reward
        self.campaign_cost = campaign_cost
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.config = {"n_owt": n_owt,
                       "lev": lev,
                       "discount_reward": discount_reward,
                       "campaign_cost": campaign_cost}
        self.struct_env = Struct_owf(self.config)
        self.date_record = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    def search(self, eval_size):
        insp_interval = np.arange(1, self.struct_env.ep_length)
        comp_inspection = np.arange(1, self.struct_env.n_agents+1)
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
        
        if not self.campaign_cost:
            camp_file = 'ref'
        else:
            camp_file = 'camp'
        path_results = "heuristics/Results"
        isExist = path.exists(path_results)
        if not isExist:
            makedirs(path_results)
        np.savez('heuristics/Results/heuristics_owf_'+ str(self.n_owt) + '_' + str(self.lev) + camp_file + '_' + self.date_record,
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
        insp_obs = np.ones((self.struct_env.n_owt,self.struct_env.lev))*2
        self.struct_env.reset()
        action = {}
        for agent in self.struct_env.agent_list:
            action[agent] = 0
        while not done_:
            action_ = action.copy()
            if (self.struct_env.time_step%insp_int)==0 and self.struct_env.time_step>0:
                probas = np.reshape(self.struct_env.damage_proba[:,:-1,:], (self.struct_env.n_agents, -1))
                pf = probas[:,-1]
                inspection_index = (-pf).argsort()[:comp_insp]
                for index in inspection_index:
                    action_[self.struct_env.agent_list[index]] = 1
            if np.sum(insp_obs) < self.struct_env.n_owt*self.struct_env.lev*2:
                inp_obs_ag = np.reshape(insp_obs[:,:-1], (self.struct_env.n_agents, -1) )
                index_repair = np.where(inp_obs_ag==1)[0]
                if len(index_repair) > 0:
                    for index in index_repair:
                        action_[self.struct_env.agent_list[index]] = 2
            [proba_, rew_, done_, insp_obs] = self.struct_env.step(action_)
            rew_total_ += rew_['agent_0']
        return rew_total_
        
