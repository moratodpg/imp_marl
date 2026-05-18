from abc import ABC, abstractmethod
from datetime import datetime
from os import makedirs

import numpy as np


class BaseHeuristicAgent(ABC):

    def __init__(self, env, config: dict, result_tag: str = ""):
        self.env = env
        self.config = config
        self.result_tag = result_tag
        self._date = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    @abstractmethod
    def episode(self, **params) -> float:
        """Run one episode and return total reward."""
        pass

    @abstractmethod
    def param_grid(self, search_params: dict) -> dict:
        """Return {param_name: 1-D array} describing the search grid."""
        pass

    def eval(self, eval_size: int, **params) -> float:
        total = sum(self.episode(**params) for _ in range(eval_size))
        mean = total / eval_size
        print(f"Eval reward: {mean:.4f}")
        return mean

    def search(self, eval_size: int, search_params: dict) -> dict:
        grid = self.param_grid(search_params)
        names = list(grid.keys())
        meshes = np.meshgrid(*[grid[k] for k in names], indexing="ij")
        flat = [m.reshape(-1) for m in meshes]
        n_points = len(flat[0])

        ret_total = np.empty(n_points)
        ret_opt, ind_opt = -1e9, 0

        for ind in range(n_points):
            params = {names[i]: flat[i][ind] for i in range(len(names))}
            mean_ret = sum(self.episode(**params) for _ in range(eval_size)) / eval_size
            ret_total[ind] = mean_ret
            if mean_ret > ret_opt:
                ret_opt = mean_ret
                ind_opt = ind
                label = " | ".join(f"{k}={flat[i][ind]}" for i, k in enumerate(names))
                print(f"  opt {ret_opt:.4f} | {label}")

        opt_params = {names[i]: flat[i][ind_opt] for i in range(len(names))}
        opt_heur = {"opt_reward_mean": ret_opt, **opt_params}
        self._save(ret_total, opt_heur)
        return opt_heur

    def _save(self, ret_total: np.ndarray, opt_heur: dict) -> None:
        makedirs("Results", exist_ok=True)
        fpath = f"Results/{self.result_tag}_{self._date}"
        np.savez(fpath, ret_total=ret_total, opt_heur=opt_heur, config=self.config)
        print(f"Saved: {fpath}.npz")

    @staticmethod
    def _component_pf(env) -> np.ndarray:
        """P(in failed state) per agent from current belief."""
        dp = env.damage_proba
        if dp.ndim == 3:  # OWF: (n_owt, lev+1, n_states)
            return dp[:, :-1, :].reshape(env.n_agents, -1)[:, -1]
        return dp[:, -1]  # struct / zayas: (n_comp, n_states)
