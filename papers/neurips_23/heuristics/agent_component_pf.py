import numpy as np

from agent_base import BaseHeuristicAgent


class ComponentPfAgent(BaseHeuristicAgent):
    """Policy driven by system and component failure probability thresholds.

    At each step:
      - Repair components where P(damaged) > repair_comp_threshold.
      - If system PF > insp_sys_threshold, inspect the top n_insp components
        by P(damaged) that are not already being repaired.

    Works with all environments (struct, owf, zayas).
    Search params: lists of exact values for each parameter.
    """

    def param_grid(self, search_params: dict) -> dict:
        return {
            "insp_sys_threshold": np.asarray(search_params["insp_sys_threshold"]),
            "repair_comp_threshold": np.asarray(search_params["repair_comp_threshold"]),
            "n_insp": np.asarray(search_params["n_insp"], dtype=int),
        }

    def episode(self, insp_sys_threshold: float, repair_comp_threshold: float, n_insp: int) -> float:
        env = self.env
        env.reset()
        total_reward = 0.0
        done = False

        while not done:
            pf_comp = self._component_pf(env)
            action = {}

            # Repair components above threshold
            for i, agent in enumerate(env.agent_list):
                action[agent] = 2 if pf_comp[i] > repair_comp_threshold else 0

            # Inspect top n_insp components if system risk is high enough
            if self._system_pf(env, pf_comp) > insp_sys_threshold:
                non_repair = [i for i in (-pf_comp).argsort() if action[env.agent_list[i]] != 2]
                for i in non_repair[: int(n_insp)]:
                    action[env.agent_list[i]] = 1

            _, rew, done, _ = env.step(action)
            total_reward += rew["agent_0"]

        return total_reward

    @staticmethod
    def _system_pf(env, pf_comp: np.ndarray) -> float:
        """Calls env.pf_sys with the correct signature for each env type."""
        if hasattr(env, "k_comp") and env.k_comp is not None:  # struct
            return env.pf_sys(pf_comp, env.k_comp)
        return env.pf_sys(pf_comp)  # zayas, owf
