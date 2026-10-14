from datetime import datetime
from os import makedirs, path

import numpy as np

from imp_marl.environments.struct_zayas import StructZayas


class HeuristicsStructZayas:
    def __init__(
        self,
        n_comp: int = 22,
        # Number of structure
        discount_reward: float = 1.0,
        # float [0,1] importance of
        # short-time reward vs long-time reward
        campaign_cost: bool = False,
        # campaign_cost = True=campaign cost taken into account
        seed=None,
    ):

        self.n_comp = n_comp
        self.discount_reward = discount_reward
        self.campaign_cost = campaign_cost
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.config = {
            "n_comp": n_comp,
            "discount_reward": discount_reward,
            "campaign_cost": campaign_cost,
        }
        self.struct_env = StructZayas(self.config)
        self.date_record = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    def search(self, eval_size):
        # insp_interval = np.arange(1, self.struct_env.ep_length)
        insp_interval = np.arange(1, self.struct_env.n_comp + 1)
        comp_inspection = np.arange(1, self.struct_env.n_comp + 1)
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
                print(
                    "opt",
                    return_heur,
                    "insp_int",
                    insp_list[ind],
                    "n_comp",
                    comp_list[ind],
                )
        self.opt_heur = {
            "opt_reward_mean": ret_opt,
            "insp_interv": insp_list[ind_opt],
            "insp_comp": comp_list[ind_opt],
        }

        if not self.campaign_cost:
            camp_file = "ref"
        else:
            camp_file = "camp"
        path_results = "heuristics/Results"
        isExist = path.exists(path_results)
        if not isExist:
            makedirs(path_results)
        np.savez(
            "heuristics/Results/heuristics_"
            + str(self.n_comp)
            + "_"
            + camp_file
            + "_"
            + self.date_record,
            ret_total=ret_total,
            opt_heur=self.opt_heur,
            config=self.config,
            seed_test=self._seed,
        )
        return self.opt_heur

    def eval(self, eval_size, insp_int, comp_insp):
        self.return_heur = 0
        for ep in range(eval_size):
            self.return_heur += self.episode(insp_int, comp_insp)
            disp_cost = self.return_heur / (ep + 1)
            if ep % 500 == 0:
                print("Reward:", disp_cost)
        self.return_heur /= eval_size
        return self.return_heur

    # def episode(self, insp_int, comp_insp):
    #     rew_total_ = 0
    #     done_ = False
    #     insp_obs = {"inspection": np.full(self.struct_env.n_comp, 2.0)}
    #     self.struct_env.reset()
    #     action = {}
    #     for agent in self.struct_env.agent_list:
    #         action[agent] = 0
    #     while not done_:
    #         action_ = action.copy()
    #         if (
    #             self.struct_env.time_step % insp_int
    #         ) == 0 and self.struct_env.time_step > 0:
    #             # pf = self.struct_env.damage_proba[:, -1]
    #             pf_ = np.zeros(self.struct_env.n_comp)
    #             for i in range(self.struct_env.n_comp):
    #                 Bplus = self.struct_env.transition_model[0, self.struct_env.d_rate[i, 0]].T.dot(self.struct_env.damage_proba[i, :])

    #                 pf_[i] = Bplus[-1]
    #             inspection_index = (-pf_).argsort()[:comp_insp]
    #             for index in inspection_index:
    #                 action_[self.struct_env.agent_list[index]] = 1
    #         if np.any(insp_obs["inspection"] == 1):
    #             index_repair = np.where(insp_obs["inspection"] == 1)[0]
    #             if len(index_repair) > 0:
    #                 for index in index_repair:
    #                     action_[self.struct_env.agent_list[index]] = 2
    #         [_, rew_, done_, insp_obs] = self.struct_env.step(action_)
    #         rew_total_ += rew_["agent_0"]
    #     return rew_total_
    
    def episode(self, insp_int=None, comp_insp=None):
        rew_total_ = 0
        done_ = False

        self.struct_env.reset()

        while not done_:
            action = self.heuristic_action(
                self.struct_env,
                comp_insp=comp_insp,
                max_repairs=insp_int,
            )

            _, rew_, done_, _ = self.struct_env.step(action)
            rew_total_ += rew_["agent_0"]

        return rew_total_
    
    def heuristic_action(self, env, comp_insp=None, max_repairs=None):
        n = env.n_comp
        action = np.zeros(n, dtype=int)

        remaining = env.ep_length - env.time_step

        # 1. Greedy repair based on system-risk reduction
        repair_scores = self.marginal_repair_scores(env)

        repair_threshold = 20.0
        if env.campaign_cost:
            repair_threshold += 5.0

        repair_candidates = np.where(repair_scores > repair_threshold)[0]
        repair_candidates = repair_candidates[np.argsort(-repair_scores[repair_candidates])]

        if max_repairs is not None:
            repair_candidates = repair_candidates[:max_repairs]

        for i in repair_candidates:
            action[i] = 2

        # 2. Inspection based on expected value of information
        if remaining > 1:
            inspection_scores = self.inspection_value_scores(env)

            # Do not inspect components already selected for repair
            inspection_scores[action == 2] = -np.inf

            inspection_candidates = np.where(inspection_scores > 0)[0]
            inspection_candidates = inspection_candidates[
                np.argsort(-inspection_scores[inspection_candidates])
            ]

            if comp_insp is not None:
                inspection_candidates = inspection_candidates[:comp_insp]

            for i in inspection_candidates:
                action[i] = 1

        return {
            env.agent_list[i]: int(action[i])
            for i in range(n)
        }
    
    def inspection_value_scores(self, env):
        B = env.damage_proba
        drate = env.d_rate
        n = env.n_comp

        repair_scores = self.marginal_repair_scores(env)
        scores = np.zeros(n)

        for i in range(n):
            # Predict belief after choosing inspection action
            p1 = env.transition_model[1, drate[i, 0]].T @ B[i]

            # inspection_model appears to have shape:
            # (n_actions, n_damage_states, n_inspection_outcomes)
            p_obs0 = np.sum(p1 * env.inspection_model[1, :, 0])
            p_obs1 = 1.0 - p_obs0

            inspection_cost = 0.2 if env.campaign_cost else 1.0

            scores[i] = p_obs1 * repair_scores[i] - inspection_cost

        return scores
    
    def marginal_repair_scores(self, env):
        B = env.damage_proba
        drate = env.d_rate
        n = env.n_comp

        # Baseline: all do nothing
        a0 = np.zeros(n, dtype=int)
        PF_base = np.zeros(n)

        for i in range(n):
            B_next_i = env.transition_model[0, drate[i, 0]].T @ B[i]
            PF_base[i] = B_next_i[-1]

        Pf_sys_base = env.pf_sys(PF_base) if n > 1 else PF_base[0]

        scores = np.zeros(n)

        for i in range(n):
            PF_repair = PF_base.copy()
            B_repair_i = env.transition_model[2, drate[i, 0]].T @ B[i]
            PF_repair[i] = B_repair_i[-1]

            Pf_sys_repair = env.pf_sys(PF_repair) if n > 1 else PF_repair[0]

            scores[i] = 100_000 * (Pf_sys_base - Pf_sys_repair)

        return scores
