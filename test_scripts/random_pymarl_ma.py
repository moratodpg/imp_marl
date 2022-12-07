import numpy as np
from struct_env.pymarl_ma_struct import PymarlMAStruct
import os
import torch as th
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.getcwd()))

    n_episode = 100

    print("test new pymarl ")
    env = PymarlMAStruct(n_comp=2,
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
    print(env_info)

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
            for k in range(env.n_agents):
                avail_actions = env.get_avail_agent_actions(k)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            # print("actions", actions)
            actions = th.from_numpy(np.array(actions))
            reward, terminated, info = env.step(actions)
            episode_reward += reward
        array_reward.append(episode_reward)
    print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
          np.min(array_reward), np.max(array_reward))

    env.close()
