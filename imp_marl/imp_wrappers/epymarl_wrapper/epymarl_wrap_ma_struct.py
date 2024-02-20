""" Wrapper for struct_env respecting the interface of EPyMARL. """

from imp_marl.imp_wrappers.pymarl_wrapper.pymarl_wrap_ma_struct import PymarlMAStruct

class EPymarlMAStruct(PymarlMAStruct):
    """
    Wrapper for Struct and Struct_owf respecting the interface of EPyMARL.
    This class inherits from PymarlMAStruct.
    It manipulates an imp_env to create all inputs for PyMARL agents.
    """

    def __init__(
        self,
        struct_type: str = "struct",
        n_comp: int = 2,
        custom_param: dict = None,
        discount_reward: float = 1.0,
        state_obs: bool = True,
        state_d_rate: bool = False,
        state_alphas: bool = False,
        obs_d_rate: bool = False,
        obs_multiple: bool = False,
        obs_all_d_rate: bool = False,
        obs_alphas: bool = False,
        env_correlation: bool = False,
        campaign_cost: bool = False,
        map_name: str = None,
        seed=None,
    ):
        """
        Initialise based on the full configuration.

        Args:
            struct_type: (str) Type of the struct env, either "struct" or "owf".
            n_comp: (int) Number of structure
            custom_param: (dict)
                struct: Number of structure required
                        {"k_comp": int} for k_comp out of n_comp
                        Default is None, meaning k_comp=n_comp-1
                 owf: Number of levels per wind turbine
                        {"lev": int}
                        Default is 3
            discount_reward: (float) Discount factor [0,1[
            state_obs: (bool) State contains the concatenation of obs
            state_d_rate: (bool) State contains the concatenation of drate
            state_alphas: (bool) State contains the concatenation of alpha
            obs_d_rate: (bool) Obs contains the drate of the agent
            obs_multiple: (bool) Obs contains the concatenation of all obs
            obs_all_d_rate: (bool) Obs contains the concatenation of all drate
            obs_alphas: (bool) Obs contains the alphas
            env_correlation: (bool) env_correlation: True=correlated, False=uncorrelated
            campaign_cost: (bool) campaign_cost = True=campaign cost taken into account
            map_name: (str) Name of the map. This is needed for establishing the interface.
            seed: (int) seed for the random number generator
        """
        super().__init__(
            struct_type=struct_type,
            n_comp=n_comp,
            custom_param=custom_param,
            discount_reward=discount_reward,
            state_obs=state_obs,
            state_d_rate=state_d_rate,
            state_alphas=state_alphas,
            obs_d_rate=obs_d_rate,
            obs_multiple=obs_multiple,
            obs_all_d_rate=obs_all_d_rate,
            obs_alphas=obs_alphas,
            env_correlation=env_correlation,
            campaign_cost=campaign_cost,
            seed=seed
        )

        self.map_name = map_name
