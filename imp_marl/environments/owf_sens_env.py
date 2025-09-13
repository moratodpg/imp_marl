""" Defines the offshore wind farm (owf) class."""

import os
import math
import numpy as np
from imp_marl.environments.imp_env import ImpEnv


class OWF_Sens(ImpEnv):
    def __init__(self, config=None):
        """offshore wind farm (owf) class.

        Attributes:
            n_owt: Number of offshore wind turbines.
            lev: Number of levels of each wind turbine.
            discount_reward: Discount factor for future rewards.
            component_costs: Costs associated with each component.
            global_costs: Costs associated with global actions.
            mobiliz_elements: Number of mobilization elements.
            pf_constraint: Constraint for the probability of failure.
            pf_sys_constraint: Constraint for the system probability of failure.
            sensor_deterioration: Deterioration rates for each sensor.

        Methods:
            reset
            step
            pf_sys
            transition
            update_sensor_conditions
        """
        if config is None:
            config = {
                "n_owt": 1,
                "lev": 3,
                "discount_reward": 1,
                "component_costs": [[0.8, 1.8, 10], [3.8, 7.8, 30]], # [insp, sensor inst, repair]
                "global_costs": [1, 100, 600], # [mobilization, corrective surplus, system failure]
                "mobiliz_elements": 5,
                "pf_constraint": 0.001, # 0.004 (ref)
                "pf_sys_constraint": 1.0,
                "sensor_deterioration": [[0.02, 0.98, 0.0], [0, 0.35, 0.65], [0.0, 0.0, 1.0]],
            }
        assert (
            "n_owt" in config
            and "lev" in config
            and "discount_reward" in config
            and "component_costs" in config
            and "global_costs" in config
            and "mobiliz_elements" in config
            and "pf_constraint" in config
            and "pf_sys_constraint" in config
            and "sensor_deterioration" in config
        ), "Missing env config"

        self.n_owt = config["n_owt"]
        self.lev = config["lev"]
        self.discount_reward = config["discount_reward"]
        self.component_costs = np.array(config["component_costs"], dtype=float)
        self.global_costs = np.array(config["global_costs"], dtype=float)
        self.mobiliz_elements = config["mobiliz_elements"]
        self.pf_constraint = config["pf_constraint"]
        self.pf_sys_constraint = config["pf_sys_constraint"]
        self.sensor_deterioration = np.array(config["sensor_deterioration"], dtype=float)

        self.n_comp = self.n_owt * self.lev
        self.n_agents = self.n_owt * (self.lev - 1) # mudline component cannot be acted upon
        self.ep_length = 20
        self.crack_conditions = 60
        self.stress_conditions = 30
        self.proba_size = self.crack_conditions * self.stress_conditions 
        self.n_obs_inspection = 2 * 30
        self.actions_per_agent = 6

        # Loading the underlying transition and inspection models
        numpy_models = np.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "pomdp_models/owf_sens_30_60_21.npz"
            )
        )

        # (n_owt, 3 levels, nstcomp cracks)
        self.initial_damage_proba = np.zeros((self.n_owt, self.lev, self.proba_size))
        self.initial_damage_proba[:] = numpy_models["belief0"]

        self.initial_sensor_condition = np.zeros((self.n_owt, self.lev, 3))
        self.initial_sensor_condition[:,:] = [0, 0, 1]  # all sensors are initially healthy

        # (3 levels, 21 det rates, 60*30 cracks_stress, 60*30 cracks_stress)
        self.transition_model = numpy_models["P"]

        # (3 sensor_conditions, 3 levels, 60 cracks, 2 observations)
        self.inspection_model = numpy_models["O"]

        # (2 actions, 3 health levels, 3 health levels)
        self.transition_sensor = numpy_models["P_sen"]

        self.agent_list = ["agent_" + str(i) for i in range(self.n_agents)]

        self.reset()

    def reset(self):
        """Resets the environment to its initial step.

        Returns:
            observations: Dictionary with the damage probability received by the agents.
        """
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        self.time_step = 0
        self.damage_proba = self.initial_damage_proba.copy()
        self.d_rate = np.zeros((self.n_owt, self.lev, 1), dtype=int)
        self.sensor_condition = self.initial_sensor_condition.copy()

        return self._get_observation()

    def step(self, action: dict):
        """Transitions the environment by one time step based on the selected actions.

        Args:
            action: Dictionary containing the actions assigned by each agent.

        Returns:
            observations: Dictionary with the damage probability received by the agents.
            rewards: Dictionary with the rewards received by the agents.
            done: Boolean indicating whether the final time step in the horizon has been reached.
        """
        action_list = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            action_list[i] = action[self.agent_list[i]]

        inspection, next_proba, next_drate, next_sensor_condition, reward_, pf_constraints_flags = self.transition(
            self.damage_proba, action_list, self.d_rate, self.sensor_condition
        )

        reward = self.discount_reward**self.time_step * reward_.item()

        rewards = {}
        for i in range(self.n_agents):
            rewards[self.agent_list[i]] = reward

        self.damage_proba = next_proba
        self.d_rate = next_drate
        self.sensor_condition = next_sensor_condition

        self.time_step += 1

        observation = self._get_observation()

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        # Info dictionary: inspection outcomes and constraint flags
        info = {"inspections": inspection}
        info.update(pf_constraints_flags)

        return observation, rewards, done, info

    def transition(self, proba, action, drate, sensor_condition):
        """Transitions the environment to the next state based on the selected actions."""
        new_proba = proba.copy()
        new_drate = drate.copy()
        new_sensor_condition = sensor_condition.copy()
        inspections = np.full((self.n_owt, self.lev), self.n_obs_inspection + 1, dtype=int)  # default "no inspection outcome" token
        pf_comp_constraint = np.zeros((self.n_owt, self.lev), dtype=bool)
        pf_sys_constraint = np.zeros(self.n_owt, dtype=bool)
        reward_sum = np.array(0.0)
        actions_count = 0

        for i in range(self.n_owt): # loop over OWTs
            # Check for system failure (observable event)
            pf_components = new_proba[i, :].reshape((self.lev, self.stress_conditions, self.crack_conditions)).sum(axis=1)[:, -1]
            pf_sys = OWF_Sens.pf_sys(pf_components)
            f_sys = np.random.choice([0, 1], size=None, replace=True, p=[1 - pf_sys, pf_sys])

            # if system failure, repair all components
            if f_sys == 1:
                new_proba[i, :, :] = self.initial_damage_proba[i, :, :].copy()
                new_drate[i, :, 0] = 0
                new_sensor_condition[i, :, :] = np.array([0, 0, 1])
                reward_sum -= self.global_costs[2] # system failure cost
                # no actions are applied from this point
                continue
            # update all component probabilities (no system failure)
            else:
                new_proba[i, :, self.crack_conditions-1:self.proba_size:self.crack_conditions] = 0
                norm = new_proba[i].sum(axis=1, keepdims=True)
                assert np.all(norm > 0), "Row with zero mass detected before normalization"
                new_proba[i, :, :] /= norm

            for j in range(self.lev - 1): # loop over levels
                action_comp = action[(self.lev - 1) * i + j]
                sensor_cond_comp = new_sensor_condition[i, j]

                if action_comp == 0:  # do-nothing
                    if sensor_cond_comp[-1] == 0:
                        new_proba[i, j], inspections[i, j] = self.observe_and_update(new_proba[i, j], 1, j)
                        new_sensor_condition[i, j, :] = self.transition_sensor[0].T @ sensor_cond_comp
                elif action_comp == 1: # do-nothing & install sensor
                    reward_sum -= self.component_costs[j, 1]; actions_count += 1
                    if sensor_cond_comp[-1] == 0:
                        new_proba[i, j], inspections[i, j] = self.observe_and_update(new_proba[i, j], 1, j) 
                    # reset sensor health
                    new_sensor_condition[i, j, :] = self.transition_sensor[1].T @ sensor_cond_comp                   
                elif action_comp == 2:  # inspect
                    reward_sum -= self.component_costs[j, 0]; actions_count += 1
                    if sensor_cond_comp[-1] == 0:
                        new_proba[i, j], inspections[i, j] = self.observe_and_update(new_proba[i, j], 2, j)
                        new_sensor_condition[i, j, :] = self.transition_sensor[0].T @ sensor_cond_comp 
                    else:
                        new_proba[i, j], inspections[i, j] = self.observe_and_update(new_proba[i, j], 0, j)
                elif action_comp == 3:  # inspect & install sensor
                    reward_sum -= (self.component_costs[j, 0] + self.component_costs[j, 1]); actions_count += 1
                    if sensor_cond_comp[-1] == 0: 
                        new_proba[i, j], inspections[i, j] = self.observe_and_update(new_proba[i, j], 2, j)
                    else:
                        new_proba[i, j], inspections[i, j] = self.observe_and_update(new_proba[i, j], 0, j)
                    # reset sensor health
                    new_sensor_condition[i, j, :] = self.transition_sensor[1].T @ sensor_cond_comp
                elif action_comp == 4:  # repair
                    reward_sum -= self.component_costs[j, 2]; actions_count += 1
                    new_proba[i, j] = self.initial_damage_proba[i, j].copy()
                    new_drate[i, j, 0] = 0
                    new_sensor_condition[i, j, :] = [0, 0, 1]
                elif action_comp == 5:  # repair & install sensor
                    reward_sum -= (self.component_costs[j, 2] + self.component_costs[j, 1]); actions_count += 1
                    new_proba[i, j] = self.initial_damage_proba[i, j].copy()
                    new_drate[i, j, 0] = 0
                    new_sensor_condition[i, j, :] = self.transition_sensor[1].T @ sensor_cond_comp

            # deterioration over one timestep
            for j in range(self.lev):
                drate_comp = new_drate[i, j, 0]
                new_proba[i, j, :] = self.transition_model[j, drate_comp].T @ (new_proba[i, j, :])
                new_drate[i, j, 0] = drate_comp + 1

            pf_components = new_proba[i, :].reshape((self.lev, self.stress_conditions, self.crack_conditions)).sum(axis=1)[:, -1]
            pf_sys = OWF_Sens.pf_sys(pf_components)

            # Component level constraints
            if self.pf_constraint is not None:
                for j in range(self.lev - 1): # mudline component cannot be acted upon
                    if pf_components[j] > self.pf_constraint:
                        pf_comp_constraint[i, j] = True
                        new_proba[i, j] = self.initial_damage_proba[i, j].copy()
                        new_drate[i, j, 0] = 0
                        new_sensor_condition[i, j, :] = [0, 0, 1]
                        reward_sum -= self.component_costs[j, 2]
                        reward_sum -= self.global_costs[1] # corrective action surplus cost
                        inspections[i, j] = self.n_obs_inspection + 1  # no inspection outcome token
            
            # System level constraint 
            if self.pf_sys_constraint is not None:
                if pf_sys > self.pf_sys_constraint:
                    pf_sys_constraint[i] = True
                    for j in range(self.lev - 1): # mudline component cannot be acted upon
                        new_proba[i, j] = self.initial_damage_proba[i, j].copy()
                        new_drate[i, j, 0] = 0
                        new_sensor_condition[i, j, :] = [0, 0, 1]
                        reward_sum -= self.component_costs[j, 2]
                        reward_sum -= self.global_costs[1] # corrective action surplus cost
                        inspections[i, j] = self.n_obs_inspection + 1  # no inspection outcome token

        # System cost (mobilization)
        if actions_count > 0 and self.global_costs[0] > 0:
            mobilization_groups = math.ceil(actions_count / self.mobiliz_elements)
            reward_sum -= self.global_costs[0] * mobilization_groups

        # Convert constraint flags to a dictionary
        pf_constraints_flags = {
            "components": pf_comp_constraint,
            "system": pf_sys_constraint
        }

        return inspections, new_proba, new_drate, new_sensor_condition, reward_sum, pf_constraints_flags

    @staticmethod
    def pf_sys(pf):
        """Computes the system failure probability (series system).
            Each wind turbine fails if any component fails.

        Args:
            pf: Numpy array with components' failure probability.

        Returns:
            PF_sys: Numpy array with the system failure probability.
        """
        surv = 1 - pf.copy()
        survC = np.prod(surv, axis=0)
        return 1 - survC
    
    # Bayesian update
    def observe_and_update(self, p, insp_type, level_comp):
        insp_prob = p @ self.inspection_model[insp_type, level_comp]
        insp_outcome = np.random.choice(range(0, self.n_obs_inspection), size=None, replace=True, p=insp_prob)
        insp_prob_unnorm = p * self.inspection_model[insp_type, level_comp, :, insp_outcome]
        new_proba = insp_prob_unnorm / np.sum(insp_prob_unnorm)
        return new_proba, insp_outcome
    
    # get observations
    def _get_observation(self):
        damage_proba_comp = OWF_Sens.reshape_observations(self.damage_proba, (self.n_agents, self.stress_conditions, self.crack_conditions))
        d_rate_comp = OWF_Sens.reshape_observations(self.d_rate, (self.n_agents, -1))
        sensor_condition_comp = OWF_Sens.reshape_observations(self.sensor_condition, (self.n_agents, -1))
        observation = {}
        for i in range(self.n_agents): 
            observation[self.agent_list[i]] = np.concatenate(
                (
                    damage_proba_comp[i].sum(axis=0),
                    damage_proba_comp[i].sum(axis=1),
                    sensor_condition_comp[i],
                    d_rate_comp[i] / self.ep_length,
                    [self.time_step / self.ep_length]
                )
            )
        # [crack condition, stress condition, sensor condition, d rate, time step]
        return observation
    
    # reshape observations
    @staticmethod
    def reshape_observations(data, new_shape):
        return np.reshape(data[:, :-1, :], new_shape)
    
