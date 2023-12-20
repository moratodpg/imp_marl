""" Demonstration of random agents with the gym wrapper. """

import numpy as np

from imp_wrappers.gymnasium.gym_wrap_sa_struct import GymSaStruct

if __name__ == "__main__":
    n_episode = 100

    env = GymSaStruct(
        struct_type="struct",
        n_comp=2,
        discount_reward=0.95,
        state_obs=True,
        state_d_rate=False,
        state_alphas=False,
        obs_d_rate=False,
        obs_multiple=False,
        obs_all_d_rate=False,
        obs_alphas=False,
        env_correlation=False,
        campaign_cost=False,
    )
    array_reward = []
    for i in range(n_episode):
        observation, info = env.reset()
        terminated = False
        episode_reward = 0
        while not terminated:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            terminated = terminated or truncated
            episode_reward += reward
        array_reward.append(episode_reward)
    print(
        n_episode,
        " mean std ",
        np.mean(array_reward),
        np.std(array_reward),
        np.min(array_reward),
        np.max(array_reward),
    )

    env.close()
