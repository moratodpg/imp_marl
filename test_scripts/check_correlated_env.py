from struct_env.pymarl_ma_struct import PymarlMAStruct
import os

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.getcwd()))
    env_1 = PymarlMAStruct(n_comp=2,
                           discount_reward=0.95,
                           k_comp=None,
                           state_obs=True,
                           state_d_rate=True,
                           state_alphas=True,
                           obs_d_rate=False,
                           obs_multiple=False,
                           obs_all_d_rate=False,
                           obs_alphas=False,
                           env_correlation=True,
                           campaign_cost=False)

    obs, state = env_1.reset()
    print("obs size", env_1.get_obs_size())
    for i in obs:
        print(len(i))
    print('state size', env_1.get_state_size())
    print(len(state))