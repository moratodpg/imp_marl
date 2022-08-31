import numpy as np
from struct_env.pymarl_ma_struct import PymarlMAStruct
from struct_env.pymarl_ma_struct_corr import PymarlMAStruct_corr
from struct_env.pymarl_sa_struct import PymarlSAStruct

if __name__ == '__main__':

    n_episode = 1000

    print("test new pymarl ")
    env_1 = PymarlMAStruct_corr()
    env_2 = PymarlMAStruct(env_type="correlated")
    env_info1 = env_1.get_env_info()
    env_info2 = env_2.get_env_info()
    print(env_info1)
    print(env_info2)

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
            obs = env_1.get_obs()
            state = env_1.get_state()

            actions = []
            for k in range(env_1.n_agents):
                avail_actions = env_1.get_avail_agent_actions(k)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            # print("actions", actions)
            reward1, terminated1, info1 = env_1.step(actions)
            reward2, terminated2, info2 = env_2.step(actions)
            episode_reward1 += reward1
            episode_reward2 += reward2
        array_reward.append(episode_reward1)
        array_reward2.append(episode_reward2)
    print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
          np.min(array_reward), np.max(array_reward))
    print(n_episode, " mean std ", np.mean(array_reward2), np.std(array_reward2),
          np.min(array_reward2), np.max(array_reward2))

