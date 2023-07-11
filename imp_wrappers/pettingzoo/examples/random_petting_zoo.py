import numpy as np

from imp_wrappers.pettingzoo.pettingzoo_wrap_struct import PettingZooStruct

if __name__ == '__main__':
    n_episode = 100

    parallel_env = PettingZooStruct(struct_type="struct",
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

        observations, infos = parallel_env.reset()
        episode_reward = 0
        while parallel_env.agents:
            # this is where you would insert your policy
            actions = {agent: parallel_env.action_space(agent).sample() for agent
                       in
                       parallel_env.agents}

            observations, rewards, terminations, truncations, infos = parallel_env.step(
                actions)
            episode_reward += rewards["agent_0"]
        array_reward.append(episode_reward)
        parallel_env.close()
    print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
          np.min(array_reward), np.max(array_reward))