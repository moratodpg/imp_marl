import numpy as np

from agent_base import BaseHeuristicAgent


class DoNothingAgent(BaseHeuristicAgent):
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
        }

    def episode(self, insp_interval: int) -> float:
        env = self.env
        env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = {}
            # Repair components above threshold
            for _, agent in enumerate(env.agent_list):
                action[agent] = 0 # Do nothing

            _, rew, done, _ = env.step(action)
            total_reward += rew["agent_0"]

        return total_reward
