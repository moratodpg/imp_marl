""" Interface for creating IMP environments. """

import os

import numpy as np

from imp_marl.environments.imp_env import ImpEnv


class Struct_Zayas(ImpEnv):
    """zayas frame system (struct_zayas) class.

    Attributes:
        n_comp: Integer indicating the number of components.
        discount_reward: Float indicating the discount factor.
        k_comp: Integer indicating the number 'k' (out of n) components in the system.
        campaign_cost: Boolean indicating whether a global campaign cost is considered in the reward model.
        ep_length: Integer indicating the number of time steps in the finite horizon.
        proba_size: Integer indicating the number of bins considered in the discretisation of the damage probability.
        n_obs_inspection: Integer indicating the number of potential outcomes resulting from an inspection.
        actions_per_agent: Integer indicating the number of actions that an agent can take.
        initial_damage_proba: Numpy array containing the initial damage probability.
        transition_model: Numpy array containing the transition model that drives the environment dynamics.
        inspection_model: Numpy array containing the inspection model.
        agent_list: Dictionary categorising the number of agents.
        time_step: Integer indicating the current time step.
        damage_proba: Numpy array contatining the current damage probability.
        d_rate: Numpy array contatining the current deterioration rate.
        observations: Dictionary listing the observations received by the agents in the Dec-POMDP.

    Methods:
        reset
        step
        pf_sys
        immediate_cost
        belief_update_uncorrelated
    Examples:
        >>> from imp_marl.environments.struct_env import Struct
        >>> import numpy as np
        >>> env = Struct()
        >>> obs = env.reset()
        >>> print(obs.keys())
        dict_keys(['agent_0', 'agent_1'])
        >>> obs["agent_0"]
        array([1.052000e-04, 5.500000e-05, 8.660000e-05, 1.261000e-04,
               2.006000e-04, 3.173000e-04, 4.853000e-04, 7.444000e-04,
               1.138400e-03, 1.783100e-03, 2.713600e-03, 4.235700e-03,
               6.473200e-03, 1.002420e-02, 1.530330e-02, 2.316180e-02,
               3.453640e-02, 5.087030e-02, 7.324320e-02, 1.008326e-01,
               1.309823e-01, 1.539425e-01, 1.567708e-01, 1.275575e-01,
               7.401660e-02, 2.583390e-02, 4.230100e-03, 2.268000e-04,
               3.200000e-06, 0.000000e+00, 0.000000e+00])
        >>> actions = {}
        >>> for agent_id in env.agent_list:
        ...     actions[agent_id] = np.random.randint(0, env.actions_per_agent)
        >>> next_obs, rewards, done, info = env.step(actions)
        >>> print(rewards.keys())
        dict_keys(['agent_0', 'agent_1'])
        >>> print(done)
        False

    """

    def __init__(self, config=None):
        """Initialises the class according to the provided config instructions.

        Args:
            config: Dictionary containing config parameters.
                Keys:
                    n_comp: Number of components.
                    discount_reward: Discount factor.
                    campaign_cost: Whether to include campaign cost in reward.
        """
        if config is None:
            config = {
                "n_comp": 2,
                "discount_reward": 1,
                "campaign_cost": False,
            }
        assert (
            "n_comp" in config
            and "discount_reward" in config
            and "campaign_cost" in config
        ), "Missing env config"

        self.n_comp = config["n_comp"]
        self.discount_reward = config["discount_reward"]
        self.campaign_cost = config["campaign_cost"]
        self.ep_length = 30
        self.proba_size = 30
        self.n_obs_inspection = 2
        self.actions_per_agent = 3

        # Loading the underlying transition and inspection models
        
        numpy_models = np.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "pomdp_models/Dr3031C10.npz",
            )
        )

        # (ncomp components, proba_size cracks)
        self.initial_damage_proba = np.zeros((self.n_comp, self.proba_size))

        self.initial_damage_proba[:, :] = numpy_models["belief0"][0, 0, :, 0]

        # (3 actions, 10 components, 31 det rates, 30 cracks, 30 cracks)
        self.transition_model = numpy_models["P"][:, 0, :, :, :]

        # (3 actions, 10 components, 30 cracks, 2 inspections)
        self.inspection_model = numpy_models["O"][:, 0, :, :]


        self.agent_list = ["agent_" + str(i) for i in range(self.n_comp)]

        self.time_step = 0
        self.damage_proba = self.initial_damage_proba
        self.d_rate = np.zeros((self.n_comp, 1), dtype=int)
        self.observations = None

        self.reset()

    def reset(self):
        """Resets the environment to its initial step.

        Returns:
            observations: Dictionary with the damage probability received by the agents.
        """
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        self.time_step = 0
        self.damage_proba = self.initial_damage_proba
        self.d_rate = np.zeros((self.n_comp, 1), dtype=int)
        self.observations = {}
        for i in range(self.n_comp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (self.damage_proba[i], [self.time_step / self.ep_length])
            )

        return self.observations

    def step(self, action: dict):
        """Transitions the environment by one time step based on the selected actions.

        Args:
            action: Dictionary containing the actions assigned by each agent.

        Returns:
            observations: Dictionary with the damage probability received by the agents.
            rewards: Dictionary with the rewards received by the agents.
            done: Boolean indicating whether the final time step in the horizon has been reached.
            inspection: Integers indicating which inspection outcomes have been collected.
        """
        action_list = np.zeros(self.n_comp, dtype=int)
        for i in range(self.n_comp):
            action_list[i] = action[self.agent_list[i]]

        inspection, next_proba, next_drate = self.belief_update(
            self.damage_proba, action_list, self.d_rate
        )

        reward_ = self.immediate_cost(
            self.damage_proba, action_list, next_proba, self.d_rate
        )
        reward = (
            self.discount_reward**self.time_step * reward_.item()
        )  # Convert float64 to float

        rewards = {}
        for i in range(self.n_comp):
            rewards[self.agent_list[i]] = reward

        self.time_step += 1

        self.observations = {}
        for i in range(self.n_comp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (next_proba[i], [self.time_step / self.ep_length])
            )

        self.damage_proba = next_proba
        self.d_rate = next_drate

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        return self.observations, rewards, done, {"inspection": inspection}

    def pf_sys(self, pf, k):
        """Computes the system failure probability pf_sys for k-out-of-n components

        Args:
            pf: Numpy array with components' failure probability.
            k: Integer indicating k (out of n) components.

        Returns:
            PF_sys: Numpy array with the system failure probability.
        """
        n = pf.size
        nk = n - k
        m = k + 1
        A = np.zeros(m + 1)
        A[1] = 1
        L = 1
        for j in range(1, n + 1):
            h = j + 1
            Rel = 1 - pf[j - 1]
            if nk < j:
                L = h - nk
            if k < j:
                A[m] = A[m] + A[k] * Rel
                h = k
            for i in range(h, L - 1, -1):
                A[i] = A[i] + (A[i - 1] - A[i]) * Rel
        PF_sys = 1 - A[m]
        return PF_sys

    def immediate_cost(self, B, a, B_, drate):
        """Computes the immediate reward (negative cost) based on current (and next) damage probability and action selected

        Args:
            B: Numpy array with current damage probability.
            a: Numpy array with actions selected.
            B_: Numpy array with the next time step damage probability.
            d_rate: Numpy array with current deterioration rates.

        Returns:
            cost_system: Float indicating the reward received.
        """
        cost_system = 0
        PF = B[:, -1]
        PF_ = B_[:, -1].copy()
        campaign_executed = False
        for i in range(self.n_comp):
            if a[i] == 1:
                cost_system += (
                    -0.2 if self.campaign_cost else -1
                )  # Individual inspection costs
                Bplus = self.transition_model[a[i], drate[i, 0]].T.dot(B[i, :])
                PF_[i] = Bplus[-1]
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True  # Campaign executed
            elif a[i] == 2:
                cost_system += -20
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True  # Campaign executed
        if self.n_comp < 2:  # single component setting
            PfSyS_ = PF_
            PfSyS = PF
        else:
            PfSyS_ = self.pf_sys(PF_, self.k_comp)
            PfSyS = self.pf_sys(PF, self.k_comp)
        if PfSyS_ < PfSyS:
            cost_system += PfSyS_ * (-10000)
        else:
            cost_system += (PfSyS_ - PfSyS) * (-10000)
        if campaign_executed:
            cost_system += -5
        return cost_system

    def belief_update(self, proba, action, drate):
        """Transitions the environment based on the current damage prob, actions selected, and current deterioration rate
            In this case, the initial damage prob are not correlated among components.

        Args:
            proba: Numpy array with current damage probability.
            action: Numpy array with actions selected.
            drate: Numpy array with current deterioration rates.

        Returns:
            inspection: Integers indicating which inspection outcomes have been collected.
            new_proba: Numpy array with the next time step damage probability.
            new_drate: Numpy array with the next time step deterioration rate.
        """
        new_proba = np.zeros((self.n_comp, self.proba_size))
        new_proba[:] = proba
        inspection = np.zeros(self.n_comp)
        new_drate = np.zeros((self.n_comp, 1), dtype=int)
        for i in range(self.n_comp):
            p1 = self.transition_model[action[i], drate[i, 0]].T.dot(
                new_proba[i, :]
            )  # environment transition

            new_proba[i, :] = p1
            # if do nothing, you update your belief without new evidences
            new_drate[i, 0] = drate[i, 0] + 1
            # At every timestep, the deterioration rate increases

            inspection[i] = 2  # ob[i] = 0 if no crack detected 1 if crack detected
            if action[i] == 1:
                ins0 = np.sum(p1 * self.inspection_model[action[i], :, 0])
                # self.observation_model = Probability to observe the crack
                ins1 = 1 - ins0

                if ins1 < 1e-5:
                    inspection[i] = 0
                else:
                    ins_dist = np.array([ins0, ins1])
                    inspection[i] = np.random.choice(
                        range(0, self.n_obs_inspection),
                        size=None,
                        replace=True,
                        p=ins_dist,
                    )
                new_proba[i, :] = (
                    p1
                    * self.inspection_model[action[i], :, int(inspection[i])]
                    / (p1.dot(self.inspection_model[action[i], :, int(inspection[i])]))
                )  # belief update
            if action[i] == 2:
                # action in b_prime has already
                # been accounted in the env transition
                new_drate[i, 0] = 0
        return inspection, new_proba, new_drate
