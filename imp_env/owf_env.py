import numpy as np

from imp_env.imp_env import ImpEnv


class Struct_owf(ImpEnv):

    def __init__(self, config=None):
        """
        :param config: (dict) A dictionary of parameters for the environment
            Keys:
            n_owt: (int) Number of wind turbines
            lev: (int) Number of levels
            discount_reward: (float) Discount factor
            campaign_cost: (bool) Whether to include campaign cost
        """
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

        self.n_owt = config["n_owt"]  # number of wind turbines
        self.lev = config["lev"]
        self.discount_reward = config["discount_reward"]
        self.campaign_cost = config["campaign_cost"]
        self.n_comp = self.n_owt*self.lev
        self.n_agents = self.n_owt*(self.lev-1)
        self.ep_length = 20  # Horizon length
        self.proba_size = 60  # Crack states (fatigue hotspot damage states)
        # Gaussian hyperparemeter states
        self.n_obs_inspection = 2
        # Total number of observations (crack detected / crack not detected)
        self.actions_per_agent = 3

        # Loading the underlying POMDP model
        numpy_models = np.load('imp_env/pomdp_models/owf6021.npz')

        # (n_owt, 3 levels, nstcomp crack states)
        self.initial_damage_proba = np.zeros((self.n_owt, self.lev, self.proba_size))

        self.initial_damage_proba[:] = numpy_models['belief0']

        # (3 actions, 3 levels, 21 det rates, 60 cracks, 60 cracks)
        self.transition_model = numpy_models['P']

        # (3 actions, 3 levels, 60 cracks, 2 observations)
        self.inspection_model = numpy_models['O']

        self.agent_list = ["agent_" + str(i) for i in range(self.n_agents)]

        self.time_step = 0
        self.damage_proba = self.initial_damage_proba

        self.d_rate = np.zeros((self.n_owt, self.lev, 1), dtype=int)
        self.observations = None # obs of the DecPomdp
        self.alphas = None  # Never used

        # Reset the environment
        self.reset()

    def reset(self):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's belief
        self.time_step = 0
        self.damage_proba = self.initial_damage_proba
        self.d_rate = np.zeros((self.n_owt, self.lev, 1), dtype=int)
        damage_proba_comp = np.reshape(self.damage_proba[:, :-1, :], (self.n_agents, -1))
        self.observations = {}

        for i in range(self.n_agents): # Shall we also add pf_sys here?
            self.observations[self.agent_list[i]] = np.concatenate(
                (damage_proba_comp[i], [self.time_step / self.ep_length]))

        return self.observations

    def step(self, action: dict):
        action_list = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            action_list[i] = action[self.agent_list[i]]

        inspection, next_proba, next_drate = \
            self.belief_update_uncorrelated(self.damage_proba, action_list,
                                            self.d_rate)

        reward_ = self.immediate_cost(self.damage_proba, action_list, next_proba,
                                      self.d_rate)
        reward = self.discount_reward ** self.time_step * reward_.item()

        rewards = {}
        for i in range(self.n_agents):
            rewards[self.agent_list[i]] = reward

        self.time_step += 1
        damage_proba_comp = np.reshape(next_proba[:,:-1,:], (self.n_agents, -1))

        self.observations = {} # Shall we also add pf_sys here?
        for i in range(self.n_agents):
            self.observations[self.agent_list[i]] = np.concatenate(
                (damage_proba_comp[i], [self.time_step / self.ep_length]))

        self.damage_proba = next_proba
        self.d_rate = next_drate

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        # info = {"belief": self.beliefs}
        return self.observations, rewards, done, None

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
                    Bplus = self.transition_model[a[(self.lev - 1) * i + j], j, drate[i, j, 0]].T.dot(B[i, j, :])
                    PF_[i] = Bplus[-1]
                    if self.campaign_cost and not campaign_executed:
                        campaign_executed = True # Campaign executed
                elif a[(self.lev-1)*i+j] == 1 and j==1:
                    cost_system += -1 if self.campaign_cost else -4
                    Bplus = self.transition_model[a[(self.lev - 1) * i + j], j, drate[i, j, 0]].T.dot(B[i, j, :])
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

    def belief_update_uncorrelated(self, proba, action, drate):
        """Bayesian belief update based on
         previous belief, current observation, and action taken"""
        # b_prime = np.zeros((self.n_comp, self.n_st_comp))
        new_proba = np.zeros((self.n_owt, self.lev, self.proba_size))
        new_proba[:] = proba
        inspection = np.zeros((self.n_owt, self.lev))
        new_drate = np.zeros((self.n_owt, self.lev, 1), dtype=int)
        for i in range(self.n_owt):
            for j in range(self.lev-1):
                p1 = self.transition_model[action[(self.lev - 1) * i + j], j, drate[i, j, 0]].T.dot(
                    new_proba[i, j, :])  # environment transition

                new_proba[i, j, :] = p1
                # if do nothing, you update your belief without new evidences
                new_drate[i, j, 0] = drate[i, j, 0] + 1
                # At every timestep, the deterioration rate increases

                inspection[i, j] = 2  # ib[o] = 0 if no crack detected 1 if crack detected
                if action[(self.lev - 1) * i + j] == 1:
                    ins0 = np.sum(p1 * self.inspection_model[action[(self.lev - 1) * i + j], j, :, 0])
                    # self.O = Probability to observe the crack
                    ins1 = 1 - ins0

                    if ins1 < 1e-5:
                        inspection[i, j] = 0
                    else:
                        ob_dist = np.array([ins0, ins1])
                        inspection[i, j] = np.random.choice(range(0, self.n_obs_inspection), size=None,
                                                    replace=True, p=ob_dist)
                    new_proba[i, j, :] = p1 * self.inspection_model[action[(self.lev - 1) * i + j], j, :, int(inspection[i, j])] / (
                        p1.dot(self.inspection_model[action[(self.lev - 1) * i + j], j, :, int(inspection[i,j])]))  # belief update
                if action[(self.lev - 1) * i + j] == 2:
                    # action in b_prime has already
                    # been accounted in the env transition
                    new_drate[i, j, 0] = 0

            ## Turbine level that cannot be inspected nor repaired
            p1 = self.transition_model[0, 2, drate[i, 2, 0]].T.dot(
                    new_proba[i, 2, :])  # environment transition
            new_proba[i, 2, :] = p1
            inspection[i, 2] = 2
            # if do nothing, you update your belief without new evidences
            new_drate[i, 2, 0] = drate[i, 2, 0] + 1
            # At every timestep, the deterioration rate increases
        return inspection, new_proba, new_drate
