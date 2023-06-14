import numpy as np

from imp_env.imp_env import ImpEnv


class Struct(ImpEnv):

    def __init__(self, config=None):
        """
        :param config: dict of config parameters, composed of:
            n_comp: number of components
            discount_reward: discount factor for reward
            k_comp: number of components required to not fail
            env_correlation: whether the damage probability is correlated or not
            campaign_cost: whether to include campaign cost in reward
        """
        if config is None:
            config = {"n_comp": 2,
                      "discount_reward": 1,
                      "k_comp": None,
                      "env_correlation": False,
                      "campaign_cost": False}
        assert "n_comp" in config and \
               "discount_reward" in config and \
               "k_comp" in config and \
               "env_correlation" in config and \
               "campaign_cost" in config, \
            "Missing env config"

        self.n_comp = config["n_comp"]
        self.discount_reward = config["discount_reward"]
        self.k_comp = self.n_comp - 1 if config["k_comp"] is None \
            else config["k_comp"]
        self.env_correlation = config["env_correlation"]
        self.campaign_cost = config["campaign_cost"]
        self.ep_length = 30  # Horizon length
        self.proba_size = 30  # Crack states (fatigue hotspot damage states)
        self.alpha_size = 80 if self.env_correlation else None  # alpha
        self.n_obs = 2  # Total number of observations (crack detected or not)
        self.actions_per_agent = 3

        # Loading the underlying POMDP model
        if not self.env_correlation:
            numpy_models = np.load('imp_env/pomdp_models/Dr3031C10.npz')
        else:
            numpy_models = np.load('imp_env/pomdp_models/Dr3031_H08.npz')

        # (ncomp components, proba_size cracks)
        self.initial_damage_proba = np.zeros((self.n_comp, self.proba_size))

        if not self.env_correlation:
            self.initial_damage_proba[:, :] = numpy_models['belief0'][0, 0, :, 0]

            # (3 actions, 10 components, 31 det rates, 30 cracks, 30 cracks)
            self.transition_model = numpy_models['P'][:, 0, :, :, :]

            # (3 actions, 10 components, 30 cracks, 2 observations)
            self.inspection_model = numpy_models['O'][:, 0, :, :]

            self.initial_damage_proba_correlated = None
            self.damage_proba_after_repair_correlated = None
            self.initial_alpha = None

        else:
            self.initial_damage_proba[:, :] = numpy_models['belief0']
            # (3 actions, 31 det rates, 30 cracks, 30 cracks)
            self.transition_model = numpy_models['P']

            # (3 actions, 30 cracks, 2 observations)
            self.inspection_model = numpy_models['O']

            self.initial_damage_proba_correlated = \
                np.zeros((self.n_comp, self.alpha_size, self.proba_size))
            self.initial_damage_proba_correlated[:, :, :] = numpy_models['belief0c']

            # conditional proba associated with a repair action
            # (80 alphas, 30 cracks)
            self.damage_proba_after_repair_correlated = numpy_models['b0cR']

            # alpha
            self.initial_alpha = numpy_models['alpha0']

        self.agent_list = ["agent_" + str(i) for i in range(self.n_comp)]

        self.time_step = 0
        self.damage_proba = self.initial_damage_proba
        self.damage_proba_correlated = self.initial_damage_proba_correlated
        self.alphas = self.initial_alpha
        self.d_rate = np.zeros((self.n_comp, 1), dtype=int)
        self.observations = None

        # Reset struct_env.
        self.reset()

    def reset(self):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's belief
        self.time_step = 0
        self.damage_proba = self.initial_damage_proba
        self.damage_proba_correlated = self.initial_damage_proba_correlated
        self.alphas = self.initial_alpha
        self.d_rate = np.zeros((self.n_comp, 1), dtype=int)
        self.observations = {}
        for i in range(self.n_comp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (self.damage_proba[i], [self.time_step / self.ep_length]))


        return self.observations

    def step(self, action: dict):
        action_list = np.zeros(self.n_comp, dtype=int)
        for i in range(self.n_comp):
            action_list[i] = action[self.agent_list[i]]

        if not self.env_correlation:
            observation_, belief_prime, drate_prime = \
                self.belief_update_uncorrelated(self.damage_proba, action_list,
                                                self.d_rate)

        else:
            observation_, belief_prime, drate_prime, bc_prime, alpha_prime = \
                self.belief_update_correlated(self.damage_proba_correlated, action_list,
                                              self.d_rate, self.alphas)

        reward_ = self.immediate_cost(self.damage_proba, action_list, belief_prime,
                                      self.d_rate)
        reward = self.discount_reward ** self.time_step * reward_.item()  # Convert float64 to float

        rewards = {}
        for i in range(self.n_comp):
            rewards[self.agent_list[i]] = reward

        self.time_step += 1

        self.observations = {}
        for i in range(self.n_comp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (belief_prime[i], [self.time_step / self.ep_length]))

        if self.env_correlation:
            self.damage_proba_correlated = bc_prime
            self.alphas = alpha_prime

        self.damage_proba = belief_prime
        self.d_rate = drate_prime

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        # info = {"belief": self.beliefs}
        return self.observations, rewards, done, observation_

    def pf_sys(self, pf, k):
        """compute pf_sys, the probability of failure of the system for k-out-of-n components"""
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
        """ immediate reward (-cost) based on current damage state and action """
        cost_system = 0
        PF = B[:, -1]
        PF_ = B_[:, -1].copy()
        campaign_executed = False
        for i in range(self.n_comp):
            if a[i] == 1:
                cost_system += -0.2 if self.campaign_cost else -1 # Individual inspection costs 
                Bplus = self.transition_model[a[i], drate[i, 0]].T.dot(B[i, :])
                PF_[i] = Bplus[-1]
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True # Campaign executed
            elif a[i] == 2:
                cost_system += - 20
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True # Campaign executed
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
        if campaign_executed: # Assign campaign cost
            cost_system += -5
        return cost_system

    def belief_update_uncorrelated(self, proba, action, drate):
        """Bayesian belief update based on
         previous belief, current observation, and action taken"""
        new_proba = np.zeros((self.n_comp, self.proba_size))
        new_proba[:] = proba
        observation = np.zeros(self.n_comp)
        new_drate = np.zeros((self.n_comp, 1), dtype=int)
        for i in range(self.n_comp):
            p1 = self.transition_model[action[i], drate[i, 0]].T.dot(
                new_proba[i, :])  # environment transition

            new_proba[i, :] = p1
            # if do nothing, you update your belief without new evidences
            new_drate[i, 0] = drate[i, 0] + 1
            # At every timestep, the deterioration rate increases

            observation[i] = 2  # ob[i] = 0 if no crack detected 1 if crack detected
            if action[i] == 1:
                Obs0 = np.sum(p1 * self.inspection_model[action[i], :, 0])
                # self.observation_model = Probability to observe the crack
                Obs1 = 1 - Obs0

                if Obs1 < 1e-5:
                    observation[i] = 0
                else:
                    ob_dist = np.array([Obs0, Obs1])
                    observation[i] = np.random.choice(range(0, self.n_obs), size=None,
                                             replace=True, p=ob_dist)
                new_proba[i, :] = p1 * self.inspection_model[action[i], :, int(observation[i])] / (
                    p1.dot(self.inspection_model[action[i], :, int(observation[i])]))  # belief update
            if action[i] == 2:
                # action in b_prime has already
                # been accounted in the env transition
                new_drate[i, 0] = 0
        return observation, new_proba, new_drate

    def belief_update_correlated(self, bc, a, drate, alpha):
        """Bayesian belief update based on previous belief,
         current observation, and action taken"""

        new_proba = np.zeros((self.n_comp, self.proba_size))
        new_proba_correlated = np.zeros((self.n_comp, self.alpha_size, self.proba_size))
        new_drate = np.zeros((self.n_comp, 1), dtype=int)
        new_alpha = alpha.copy()
        observation = np.zeros(self.n_comp)
        for i in range(self.n_comp):
            p1 = bc[i, :, :].dot(
                self.transition_model[a[i], drate[i, 0]])  # environment transition
            new_proba_correlated[i, :, :] = p1
            new_drate[i, 0] = drate[i, 0] + 1

            if a[i] == 1:
                Obs0 = np.sum(alpha[:].dot(p1) * self.inspection_model[a[i], :, 0])
                Obs1 = 1 - Obs0
                if Obs1 < 1e-5:
                    observation[i] = 0
                else:
                    ob_dist = np.array([Obs0, Obs1])
                    observation[i] = np.random.choice(range(0, self.n_obs), size=None,
                                             replace=True, p=ob_dist)
                pInsp = p1 * self.inspection_model[a[i], :, int(observation[i])]  # belief update
                likAlpha = np.sum(pInsp, axis=1)  # likelihood insp alpha
                normBel = np.tile(likAlpha, self.proba_size).reshape(
                    self.alpha_size, self.proba_size,
                    order='F')  # normalization constant
                new_proba_correlated[i, :, :] = pInsp / normBel
                alpha_curr = likAlpha * new_alpha
                new_alpha = alpha_curr / np.sum(alpha_curr)

            if a[i] == 2:
                new_proba_correlated[i, :, :] = self.damage_proba_after_repair_correlated
                new_drate[i, 0] = 0

        for i in range(self.n_comp):
            new_proba[i, :] = new_alpha.dot(
                new_proba_correlated[i, :, :])  # Belief (marginalize out alpha)
        return observation, new_proba, new_drate, new_proba_correlated, new_alpha
