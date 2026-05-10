""" Interface for creating IMP environments. """

import os

import numpy as np

from imp_marl.environments.imp_env import ImpEnv

ZAYAS_TOPOLOGY = [
    [0, 1],    # element 0: hotspots 0, 1
    [2, 3],    # element 1: hotspots 2, 3
    [4],       # element 2: hotspot 4 only
    [5],       # element 3
    [6, 7],    # element 4
    [8, 9],    # element 5
    [10],      # element 6
    [11],      # element 7
    [12, 13],  # element 8
    [14, 15],  # element 9
    [16, 17],  # element 10
    [18, 19],  # element 11
    [20, 21],  # element 12
]


class StructZayas(ImpEnv):
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
                "n_comp": 22,
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
                "pomdp_models/zayas_model.npz",
            )
        )

        # (ncomp components, proba_size cracks)
        self.initial_damage_proba = np.zeros((self.n_comp, self.proba_size))

        self.initial_damage_proba[:, :] = numpy_models["belief_0"]

        # (3 actions, n_comp components, 31 det rates, 30 cracks, 30 cracks)
        self.transition_model = numpy_models["P"]

        # (3 actions, n_comp components, 30 cracks, 2 inspections)
        self.inspection_model = numpy_models["O"]

        self.sys_surv_cond = numpy_models["surv_sys_cond"]

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
    
    def connect_zayas(self, pf_hotspot, topology=ZAYAS_TOPOLOGY):
        """Compute element failure probabilities from hotspot failure probs.
        
        Each element is a series system of its hotspots:
        pf_elem = 1 - prod(1 - pf_hotspot[j]) for j in element's hotspots.
        """
        surv_comp = 1.0 - pf_hotspot
        surv_elem = np.array([
            np.prod(surv_comp[idx]) for idx in topology
        ])
        return 1.0 - surv_elem

    def elem_state(self, pf_elem):
        """Joint probability vector over all 2^n element state combinations.
        
        Assumes elements are independent. Returns a vector of length 2^n,
        where entry k corresponds to the binary state (failed/survived)
        encoded by the binary representation of k.
        
        Equivalent to the Kronecker product of [pf_i, 1-pf_i] vectors.
        """
        n = len(pf_elem)
        q = np.array([pf_elem[0], 1.0 - pf_elem[0]])
        for i in range(1, n):
            q = np.kron(q, [pf_elem[i], 1.0 - pf_elem[i]])
        return q
    
    def pf_sys(self, pf_hotspot, topology=ZAYAS_TOPOLOGY):
        """Compute system failure probability from hotspot failure probabilities.
        
        Parameters
        ----------
        pf_hotspot : array of shape (n_hotspots,)
            Failure probability of each hotspot.
        surv_sys_cond : array of shape (2^n_elem,)
            Conditional system survival probability for each element state combo.
        topology : list of lists
            Mapping from elements to hotspot indices.
        
        Returns
        -------
        float
            System failure probability.
        """
        pf_elem = self.connect_zayas(pf_hotspot, topology)
        q = self.elem_state(pf_elem)
        return 1 - np.dot(self.sys_surv_cond, q)

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
            # PfSyS = PF
        else:
            PfSyS_ = self.pf_sys(PF_)
            # PfSyS = self.pf_sys(PF)
        # if PfSyS_ < PfSyS:
        cost_system += PfSyS_ * (-100_000)
        # else:
        #     cost_system += (PfSyS_ - PfSyS) * (-100_000)
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
