""" Demonstration of random agents with the gym wrapper. """

from imp_marl.imp_wrappers.gym.gym_wrap_sa_struct import GymSaStruct
import numpy as np

if __name__ == '__main__':
    n_episode = 100

    env = GymSaStruct(struct_type="struct",
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
    array_reward = []
    for i in range(n_episode):
        observation = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
        array_reward.append(episode_reward)

    print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
          np.min(array_reward), np.max(array_reward))
    env.close()
