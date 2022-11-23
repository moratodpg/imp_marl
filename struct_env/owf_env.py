import numpy as np


class Struct_owf:

    def __init__(self, config=None):
        if config is None:
            config = {"n_owt": 2,
                      "lev": 3,
                      "discount_reward": 1,
                      "campaign_cost": False}
        assert "n_owt" in config and \
               "lev" in config and \
               "discount_reward" in config and \
               "campaign_cost" in config, \
            "Missing env config"

        self.n_owt = config["n_owt"]
        self.lev = config["lev"]
        self.discount_reward = config["discount_reward"]
        self.campaign_cost = config["campaign_cost"]
        self.time = 0
        self.n_comp = self.n_owt*self.lev
        self.n_agents = self.n_owt*(self.lev-1)
        self.ep_length = 20  # Horizon length
        self.n_st_comp = 60  # Crack states (fatigue hotspot damage states)
        # Gaussian hyperparemeter states
        self.n_obs = 2
        # Total number of observations (crack detected / crack not detected)
        self.actions_per_agent = 3

        # Uncorrelated obs = 60 per agent + 1 timestep
        self.obs_per_agent_multi = None  # Todo: check
        self.obs_total_single = None  # Todo: check used in gym env

        ### Loading the underlying POMDP model ###

        drmodel = np.load('pomdp_models/owf6021.npz')

        # (n_owt, 3 levels, nstcomp crack states)
        self.belief0 = np.zeros((self.n_owt, self.lev, self.n_st_comp))

        self.belief0[:] = drmodel['belief0']

        # (3 actions, 3 levels, 21 det rates, 60 cracks, 60 cracks)
        self.P = drmodel['P']

        # (3 actions, 3 levels, 60 cracks, 2 observations)
        self.O = drmodel['O']

        self.agent_list = ["agent_" + str(i) for i in range(self.n_agents)]

        self.time_step = 0
        self.beliefs = self.belief0

        self.d_rate = np.zeros((self.n_owt, self.lev, 1), dtype=int)
        self.observations = None

        # Reset struct_env.
        self.reset()

    def reset(self):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's belief
        self.time_step = 0
        self.beliefs = self.belief0
        self.d_rate = np.zeros((self.n_owt, self.lev, 1), dtype=int)
        beliefs_comp = np.reshape(self.beliefs[:,:-1,:], (self.n_agents, -1))
        self.observations = {}

        for i in range(self.n_agents): # Shall we also add pf_sys here?
            self.observations[self.agent_list[i]] = np.concatenate(
                (beliefs_comp[i], [self.time_step / self.ep_length]))

        return self.observations

    def step(self, action: dict):
        action_ = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            action_[i] = action[self.agent_list[i]]

        observation_, belief_prime, drate_prime = \
            self.belief_update_uncorrelated(self.beliefs, action_,
                                            self.d_rate)

        reward_ = self.immediate_cost(self.beliefs, action_, belief_prime,
                                      self.d_rate)
        reward = self.discount_reward ** self.time_step * reward_.item()  # Convert float64 to float

        rewards = {}
        for i in range(self.n_agents):
            rewards[self.agent_list[i]] = reward

        self.time_step += 1
        beliefs_comp = np.reshape(belief_prime[:,:-1,:], (self.n_agents, -1))

        self.observations = {} # Shall we also add pf_sys here?
        for i in range(self.n_agents):
            self.observations[self.agent_list[i]] = np.concatenate(
                (beliefs_comp[i], [self.time_step / self.ep_length]))

        self.beliefs = belief_prime
        self.d_rate = drate_prime

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        # info = {"belief": self.beliefs}
        return self.observations, rewards, done

    def pf_sys(self, pf): # immediate reward (-cost), based on current damage state and action#
    #  	n = pf.size
        pfSys = np.zeros(self.n_owt)
        surv = 1 - pf.copy()
        #failsys = np.zeros((nwtb,2))
        for i in range(self.n_owt):
            survC = surv[i,0]*surv[i,1]*surv[i,2]
            pfSys[i] = 1 - survC
        return pfSys

    def immediate_cost(self, B, a, B_, drate):
        """ immediate reward (-cost),
         based on current damage state and action """
        cost_system = 0
        PF = B[:, :, -1].copy()
        PF_ = B_[:, :, -1].copy()
        campaign_executed = False
        for i in range(self.n_owt):
            for j in range(self.lev-1):
                if a[(self.lev-1)*i+j] == 1 and j==0:
                    cost_system += -0.2 if self.campaign_cost else -1
                    Bplus = self.P[a[(self.lev-1)*i+j], j, drate[i, j, 0]].T.dot(B[i, j, :])
                    PF_[i] = Bplus[-1]
                    if self.campaign_cost and not campaign_executed:
                        campaign_executed = True # Campaign executed
                elif a[(self.lev-1)*i+j] == 1 and j==1:
                    cost_system += -1 if self.campaign_cost else -4
                    Bplus = self.P[a[(self.lev-1)*i+j], j, drate[i, j, 0]].T.dot(B[i, j, :])
                    PF_[i] = Bplus[-1]
                    if self.campaign_cost and not campaign_executed:
                        campaign_executed = True # Campaign executed                    
                elif a[(self.lev-1)*i+j] == 2 and j==0:
                    cost_system += - 10
                elif a[(self.lev-1)*i+j] == 2 and j==1:
                    cost_system += - 30
                    if self.campaign_cost and not campaign_executed:
                        campaign_executed = True # Campaign executed
        PfSyS = self.pf_sys(PF)
        PfSyS_ = self.pf_sys(PF_)
        for i in range(self.n_owt):
            if PfSyS_[i] < PfSyS[i]:
                cost_system += PfSyS_[i] * (-1000)
            else:
                cost_system += (PfSyS_[i] - PfSyS[i]) * (-1000)
        if campaign_executed: # Assign campaign cost
            cost_system += -5

        return cost_system

    def belief_update_uncorrelated(self, b, a, drate):
        """Bayesian belief update based on
         previous belief, current observation, and action taken"""
        # b_prime = np.zeros((self.n_comp, self.n_st_comp))
        b_prime = np.zeros((self.n_owt, self.lev, self.n_st_comp))
        b_prime[:] = b
        ob = np.zeros((self.n_owt, self.lev))
        # drate_prime = np.zeros((self.n_comp, 1), dtype=int)
        drate_prime = np.zeros((self.n_owt, self.lev, 1), dtype=int)
        # for i in range(self.n_comp):
        for i in range(self.n_owt):
            for j in range(self.lev-1):
                p1 = self.P[a[(self.lev-1)*i+j], j, drate[i, j, 0]].T.dot(
                    b_prime[i, j, :])  # environment transition

                b_prime[i, j, :] = p1
                # if do nothing, you update your belief without new evidences
                drate_prime[i, j, 0] = drate[i, j, 0] + 1
                # At every timestep, the deterioration rate increases

                ob[i, j] = 2  # ib[o] = 0 if no crack detected 1 if crack detected
                if a[i+j] == 1:
                    Obs0 = np.sum(p1 * self.O[a[(self.lev-1)*i+j], j, :, 0])
                    # self.O = Probability to observe the crack
                    Obs1 = 1 - Obs0

                    if Obs1 < 1e-5:
                        ob[i, j] = 0
                    else:
                        ob_dist = np.array([Obs0, Obs1])
                        ob[i, j] = np.random.choice(range(0, self.n_obs), size=None,
                                                replace=True, p=ob_dist)
                    b_prime[i, j, :] = p1 * self.O[a[(self.lev-1)*i+j], j, :, int(ob[i, j])] / (
                        p1.dot(self.O[a[(self.lev-1)*i+j], j, :, int(ob[i,j])]))  # belief update
                if a[i] == 2:
                    # action in b_prime has already
                    # been accounted in the env transition
                    drate_prime[i, j, 0] = 0

            ## Turbine level that cannot be inspected nor repaired
            p1 = self.P[a[0], 2, drate[i, 2, 0]].T.dot(
                    b_prime[i, 2, :])  # environment transition
            b_prime[i, 2, :] = p1
            # if do nothing, you update your belief without new evidences
            drate_prime[i, 2, 0] = drate[i, 2, 0] + 1
            # At every timestep, the deterioration rate increases
        return ob, b_prime, drate_prime