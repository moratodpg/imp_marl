""" Test the two PyMarl wrapper with the same struct parameters. """

import numpy as np
from imp_wrappers.pymarl_wrapper.pymarl_wrap_ma_struct import PymarlMAStruct
from imp_wrappers.pymarl_wrapper.pymarl_wrap_sa_struct import PymarlSAStruct

if __name__ == '__main__':

    n_episode = 10
    env_1 = PymarlMAStruct(struct_type="struct",
                           n_comp=2,
                           discount_reward=.95,
                           state_obs=True,
                           state_d_rate=True,
                           state_alphas=True,
                           obs_d_rate=False,
                           obs_multiple=False,
                           obs_all_d_rate=False,
                           obs_alphas=False,
                           env_correlation=True,
                           campaign_cost=True)
    env_2 = PymarlSAStruct(struct_type="struct",
                           n_comp=2,
                           discount_reward=.95,
                           state_obs=True,
                           state_d_rate=True,
                           state_alphas=True,
                           obs_d_rate=False,
                           obs_multiple=False,
                           obs_all_d_rate=False,
                           obs_alphas=False,
                           env_correlation=True,
                           campaign_cost=True)
    env_info1 = env_1.get_env_info()
    env_info2 = env_2.get_env_info()

    n_actions = env_info1["n_actions"]
    n_agents = env_info1["n_agents"]
    n_actions2 = env_info2["n_actions"]
    n_agents2 = env_info2["n_agents"]

    array_reward = []
    array_reward2 = []

    for e in range(n_episode):
        env_1.reset()
        env_2.reset()

        terminated1 = False
        episode_reward1 = 0
        episode_reward2 = 0

        while not terminated1:
            obs1 = env_1.get_obs()
            obs2 = env_2.get_obs()
            state1 = env_1.get_state()
            state2 = env_2.get_state()

            actions = []
            for k in range(env_1.n_agents):
                avail_actions = env_1.get_avail_agent_actions(k)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            reward1, terminated1, info1 = env_1.step(actions)
            for k, v in env_2.convert_action_dict.items():
                if np.all(v == actions):
                    actions_sarl = [k]
            reward2, terminated2, info2 = env_2.step(actions_sarl)
            episode_reward1 += reward1
            episode_reward2 += reward2
        array_reward.append(episode_reward1)
        array_reward2.append(episode_reward2)
    print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
          np.min(array_reward), np.max(array_reward))
    print(n_episode, " mean std ", np.mean(array_reward2),
          np.std(array_reward2),
          np.min(array_reward2), np.max(array_reward2))
