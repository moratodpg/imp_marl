import numpy as np

from agent_base import BaseHeuristicAgent


class IntervalAgent(BaseHeuristicAgent):
    """Inspect top-k components at fixed intervals; repair observed-damaged components.

    Works with struct and owf environments.
    Search params: insp_interval [start, stop), comp_inspection [start, stop).
    """

    def param_grid(self, search_params: dict) -> dict:
        ii = search_params.get("insp_interval", [1, self.env.ep_length])
        ci = search_params.get("comp_inspection", [1, len(self.env.agent_list) + 1])
        return {
            "insp_interval": np.arange(ii[0], ii[1], dtype=int),
            "comp_inspection": np.arange(ci[0], ci[1], dtype=int),
        }

    def episode(self, insp_interval: int, comp_inspection: int) -> float:
        env = self.env
        env.reset()
        last_obs = None
        total_reward = 0.0
        done = False

        while not done:
            action = {a: 0 for a in env.agent_list}

            # Inspect top-k components by P(damaged) every insp_interval steps
            if env.time_step > 0 and env.time_step % int(insp_interval) == 0:
                pf = self._component_pf(env)
                for idx in (-pf).argsort()[: int(comp_inspection)]:
                    action[env.agent_list[idx]] = 1

            # Repair any component whose last inspection revealed damage
            for idx in self._repair_indices(env, last_obs):
                action[env.agent_list[idx]] = 2

            result = env.step(action)
            last_obs, rew, done = result[3], result[1], result[2]
            total_reward += rew["agent_0"]

        return total_reward

    @staticmethod
    def _repair_indices(env, last_obs):
        if last_obs is None:
            return []
        # zayas returns {"inspection": array}; struct/owf return a plain array
        if isinstance(last_obs, dict):
            obs = np.asarray(last_obs["inspection"])
        else:
            obs = np.atleast_1d(np.asarray(last_obs))
        if obs.ndim == 2:  # OWF: shape (n_owt, lev)
            flat = obs[:, :-1].reshape(env.n_agents, -1)
            return np.where(flat == 1)[0].tolist()
        return np.where(obs == 1)[0].tolist()  # struct / zayas: shape (n_comp,)
