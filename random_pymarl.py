import numpy as np
from struct_env.pymarl_ma_struct import PymarlMAStruct
from struct_env.pymarl_sa_struct import PymarlSAStruct

if __name__ == '__main__':

    n_episode = 1

    print("test new pymarl ")
    env = PymarlSAStruct(state_config="all", discount_reward=1)

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
           #print("actions", actions)
            reward, terminated, info = env.step(actions)
            episode_reward += reward
        array_reward.append(episode_reward)
    print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
          np.min(array_reward), np.max(array_reward))

    env.close()