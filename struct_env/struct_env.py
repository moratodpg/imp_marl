import numpy as np


class Struct:

    def __init__(self, config=None):
        config = config or {"config": {"components": 2}}
        # Number of components #
        self.ncomp = config['config']["components"]
        self.time = 0
        self.ep_length = 30
        self.nstcomp = 30  # What is this?
        self.nobs = 2  # What is this?
        self.actions_per_agent = 3

        # obs = 30 per agent + 1 timestep
        self.obs_per_agent_multi = 31
        self.obs_total_single = 30 * self.ncomp + 1

        ### Loading the underlying POMDP model ###
        drmodel = np.load('pomdp_models/Dr3031C10.npz')

        # (10 components, 30 crack states)
        self.belief0 = drmodel['belief0'][0, 0:self.ncomp, :, 0]

        # (3 actions, 10 components, 31 det rates, 30 cracks, 30 cracks)
        self.P = drmodel['P'][:, 0:self.ncomp, :, :, :]

        # (3 actions, 10 components, 30 cracks, 2 observations)
        self.O = drmodel['O'][:, 0:self.ncomp, :, :]

        self.agent_list = ["agent_" + str(i) for i in range(self.ncomp)]

        self.time_step = 0
        self.agent_belief = self.belief0
        self.drate = np.zeros((self.ncomp, 1), dtype=int)

        # Reset struct_env.
        self.reset()

    def reset(self):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's belief
        self.time_step = 0
        self.agent_belief = self.belief0
        self.drate = np.zeros((self.ncomp, 1), dtype=int)
        observations = {}
        for i in range(self.ncomp):
            observations[self.agent_list[i]] = np.concatenate(
                (self.agent_belief[i], [self.time_step / 30]))
        return observations

    def step(self, action: dict):
        action_ = np.zeros(self.ncomp, dtype=int)
        for i in range(self.ncomp):
            action_[i] = action[self.agent_list[i]]

        observation_, belief_prime, drate_prime = self.belief_update(
            self.agent_belief, action_, self.drate)

        self.time_step += 1

        observations = {}
        for i in range(self.ncomp):
            observations[self.agent_list[i]] = np.concatenate(
                (belief_prime[i], [self.time_step / 30]))

        reward_ = self.immediate_cost(self.agent_belief, action_, belief_prime,
                                      self.drate)
        reward = reward_.item()  # Convert float64 to float

        rewards = {}
        for i in range(self.ncomp):
            rewards[self.agent_list[i]] = reward

        self.agent_belief = belief_prime
        self.drate = drate_prime

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        # info = {"belief": self.agent_belief}
        return observations, rewards, done

    def pf_sys(self, pf, k):  # compute pf_sys for k-out-of-n components
        n = pf.size
        # k = ncomp-1
        PF_sys = np.zeros(1)
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

    def immediate_cost(self, B, a, B_,
                       drate):  # immediate reward (-cost), based on current damage state and action#
        cost_system = 0
        PF = np.zeros((1, 1))
        PF = B[:, -1]
        PF_ = np.zeros((1, 1))
        PF_ = B_[:, -1].copy()
        for i in range(self.ncomp):
            if a[i] == 1:
                cost_system += -1
                Bplus = self.P[a[i], i, drate[i, 0]].T.dot(B[i, :])
                PF_[i] = Bplus[-1]
            elif a[i] == 2:
                cost_system += - 20
        if self.ncomp < 2:  # single component setting
            PfSyS_ = PF_
            PfSyS = PF
        else:
            PfSyS_ = self.pf_sys(PF_, self.ncomp - 1)
            PfSyS = self.pf_sys(PF, self.ncomp - 1)
        if PfSyS_ < PfSyS:
            cost_system += PfSyS_ * (-10000)
        else:
            cost_system += (PfSyS_ - PfSyS) * (-10000)
        return cost_system

    def belief_update(self, b, a,
                      drate):  # Bayesian belief update based on previous belief, current observation, and action taken
        b_prime = np.zeros((self.ncomp, self.nstcomp))
        b_prime[:] = b
        ob = np.zeros(self.ncomp)
        drate_prime = np.zeros((self.ncomp, 1), dtype=int)
        for i in range(self.ncomp):
            p1 = self.P[a[i], i, drate[i, 0]].T.dot(
                b_prime[i, :])  # environment transition
            b_prime[i, :] = p1
            drate_prime[i, 0] = drate[i, 0] + 1
            ob[i] = 2
            if a[i] == 1:
                Obs0 = np.sum(p1 * self.O[a[i], i, :, 0])
                Obs1 = 1 - Obs0
                if Obs1 < 1e-5:
                    ob[i] = 0
                else:
                    ob_dist = np.array([Obs0, Obs1])
                    ob[i] = np.random.choice(range(0, self.nobs), size=None,
                                             replace=True, p=ob_dist)
                b_prime[i, :] = p1 * self.O[a[i], i, :, int(ob[i])] / (
                    p1.dot(self.O[a[i], i, :, int(ob[i])]))  # belief update
            if a[i] == 2:
                drate_prime[i, 0] = 0
        return ob, b_prime, drate_prime
