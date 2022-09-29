import itertools

import gym
import numpy as np

from struct_env.pymarl_ma_struct import PymarlMAStruct
from struct_env.struct_env import Struct

if __name__ == '__main__':

    n_episode = 10
    action_MA = {"agent_0": 0,  # action_space.sample(),
                 "agent_1": 0}  # action_space.sample()}
    # {0: array([0, 0]), 1: array([0, 1]), 2: array([0, 2]), 3: array([1, 0]),
    # 4: array([1, 1]), 5: array([1, 2]), 6: array([2, 0]), 7: array([2, 1]),
    # 8: array([2, 2])}
    # action_SA = 0
    # print("test Struct")
    # env = Struct()
    # action_space = gym.spaces.Discrete(env.actions_per_agent)
    # array_reward = []
    # for _ in range(n_episode):
    #     env.reset()
    #     done = False
    #     id = 0
    #     rew_tot = 0
    #     while not done:
    #         observations, rewards, done = env.step(action_MA)
    #         # print(id, 'action:', action, 'reward:', rewards, done)
    #         id += 1
    #         rew_tot += rewards["agent_1"]
    #     array_reward.append(rew_tot)
    #
    # print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
    #       np.min(array_reward), np.max(array_reward))
    # print()
    # print("test GYM SA")
    # env_sa = GymSaStruct()
    # array_reward = []
    # for _ in range(n_episode):
    #     env_sa.reset()
    #     done = False
    #     id = 0
    #     rew_tot = 0
    #     while not done:
    #         observation, reward, done, info = env_sa.step(action_SA)  # env_sa.action_space.sample())
    #         # print(id, 'action: ', action, 'reward:', reward, done)
    #         rew_tot += reward
    #         id += 1
    #     # print(rew_tot)
    #     array_reward.append(rew_tot)
    # print(n_episode, "mean std ", np.mean(array_reward), np.std(array_reward),
    #       np.min(array_reward), np.max(array_reward))
    # print()
    # print("test old gym SA")
    # env_sa_old = StructSA()
    # array_reward = []
    # for _ in range(n_episode):
    #     env_sa_old.reset()
    #     done = False
    #     id = 0
    #     rew_tot = 0
    #     while not done:
    #         observation, reward, done, info = env_sa_old.step(action_SA)  # env_sa.action_space.sample())
    #         # print(id, 'action: ', action, 'reward:', reward, done)
    #         rew_tot += reward
    #         id += 1
    #     # print(rew_tot)
    #     array_reward.append(rew_tot)
    # print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
    #       np.min(array_reward), np.max(array_reward))
    # print()
    # print("test old gym ray")
    # env_ma_old = StructMA()
    # array_reward = []
    # for _ in range(n_episode):
    #     env_ma_old.reset()
    #     done = False
    #     id = 0
    #     rew_tot = 0
    #     while not done:
    #         observation, reward, done, info = env_ma_old.step(
    #             action_MA)  # env_sa.action_space.sample())
    #         # print(id, 'action: ', action, 'reward:', reward, done)
    #         rew_tot += reward["agent_1"]
    #         done = done["__all__"]
    #         id += 1
    #         # print(id, rew_tot)
    #     array_reward.append(rew_tot)
    # print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
    #       np.min(array_reward), np.max(array_reward))
    #
    # print()
    # print("test new gym ray")
    # env_ma_old = RayMaStruct()
    # array_reward = []
    # for _ in range(n_episode):
    #     env_ma_old.reset()
    #     done = False
    #     id = 0
    #     rew_tot = 0
    #     while not done:
    #         observation, reward, done, info = env_ma_old.step(
    #             action_MA)  # env_sa.action_space.sample())
    #         # print(id, 'action: ', action, 'reward:', reward, done)
    #         rew_tot += reward["agent_1"]
    #         done = done["__all__"]
    #         id += 1
    #         # print(id, rew_tot)
    #     array_reward.append(rew_tot)
    # print(n_episode, " mean std ", np.mean(array_reward), np.std(array_reward),
    #       np.min(array_reward), np.max(array_reward))
    #
    # print()
    print("test new pymarl ")
    env = PymarlMAStruct()

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
            for k, v in action_MA.items():
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