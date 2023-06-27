""" Demonstration of the PymarlMAStruct wrapper with random actions for owf. """

import numpy as np
from imp_wrappers.pymarl_wrapper.pymarl_wrap_sa_struct import PymarlSAStruct
if __name__ == '__main__':

    n_episode = 100

    env = PymarlSAStruct(struct_type="owf",
                         n_comp=2,
                         discount_reward=.95,
                         state_obs=True,
                         state_d_rate=False,
                         state_alphas=False,
                         obs_d_rate=False,
                         obs_multiple=False,
                         obs_all_d_rate=False,
                         obs_alphas=False,
                         env_correlation=False,
                         campaign_cost=False)

    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    array_reward = []

    for e in range(n_episode):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()

            actions = []
            avail_actions = env.get_avail_agent_actions(0)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_ind)
            actions.append(action)
            reward, terminated, info = env.step(actions)
            episode_reward += reward
        array_reward.append(episode_reward)
    print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
          np.min(array_reward), np.max(array_reward))

    env.close()
