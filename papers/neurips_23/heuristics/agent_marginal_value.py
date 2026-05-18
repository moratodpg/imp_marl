import numpy as np

from agent_base import BaseHeuristicAgent


class MarginalValueAgent(BaseHeuristicAgent):
    """Two-phase policy: greedy repair by system-PF reduction, then inspection by expected value.

    Designed for the Zayas environment (requires env.pf_sys, env.transition_model,
    env.inspection_model, env.d_rate).

    Search params: max_repairs [start, stop), comp_inspection [start, stop).
    Agent params: repair_threshold (default 20.0) — minimum marginal score to trigger repair.
    """

    def __init__(self, env, config: dict, result_tag: str = "", repair_threshold: float = 20.0):
        super().__init__(env, config, result_tag)
        self.repair_threshold = repair_threshold

    def param_grid(self, search_params: dict) -> dict:
        n = self.env.n_comp
        mr = search_params.get("max_repairs", [1, n + 1])
        ci = search_params.get("comp_inspection", [1, n + 1])
        return {
            "max_repairs": np.arange(mr[0], mr[1], dtype=int),
            "comp_inspection": np.arange(ci[0], ci[1], dtype=int),
        }

    def episode(self, max_repairs: int, comp_inspection: int) -> float:
        env = self.env
        env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self._get_action(env, int(comp_inspection), int(max_repairs))
            _, rew, done, _ = env.step(action)
            total_reward += rew["agent_0"]

        return total_reward

    def _get_action(self, env, comp_inspection: int, max_repairs: int) -> dict:
        n = env.n_comp
        action = np.zeros(n, dtype=int)
        remaining = env.ep_length - env.time_step

        # Phase 1: repair components with highest marginal system-PF reduction
        repair_scores = self._marginal_repair_scores(env)
        threshold = self.repair_threshold + (5.0 if env.campaign_cost else 0.0)
        candidates = np.where(repair_scores > threshold)[0]
        candidates = candidates[np.argsort(-repair_scores[candidates])][:max_repairs]
        action[candidates] = 2

        # Phase 2: inspect components with highest expected information value
        if remaining > 1:
            insp_scores = self._inspection_value_scores(env, repair_scores)
            insp_scores[action == 2] = -np.inf  # skip already-repaired
            candidates = np.where(insp_scores > 0)[0]
            candidates = candidates[np.argsort(-insp_scores[candidates])][:comp_inspection]
            action[candidates] = 1

        return {env.agent_list[i]: int(action[i]) for i in range(n)}

    def _marginal_repair_scores(self, env) -> np.ndarray:
        B, drate, n = env.damage_proba, env.d_rate, env.n_comp
        pf_base = np.array(
            [(env.transition_model[0, drate[i, 0]].T @ B[i])[-1] for i in range(n)]
        )
        pf_sys_base = env.pf_sys(pf_base) if n > 1 else pf_base[0]

        scores = np.zeros(n)
        for i in range(n):
            pf_rep = pf_base.copy()
            pf_rep[i] = (env.transition_model[2, drate[i, 0]].T @ B[i])[-1]
            pf_sys_rep = env.pf_sys(pf_rep) if n > 1 else pf_rep[0]
            scores[i] = 100_000 * (pf_sys_base - pf_sys_rep)
        return scores

    def _inspection_value_scores(self, env, repair_scores: np.ndarray) -> np.ndarray:
        B, drate, n = env.damage_proba, env.d_rate, env.n_comp
        insp_cost = 0.2 if env.campaign_cost else 1.0
        scores = np.zeros(n)
        for i in range(n):
            p1 = env.transition_model[1, drate[i, 0]].T @ B[i]
            p_obs0 = np.sum(p1 * env.inspection_model[1, :, 0])
            p_obs1 = 1.0 - p_obs0
            scores[i] = p_obs1 * repair_scores[i] - insp_cost
        return scores
