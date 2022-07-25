import itertools

import gym
from gym import spaces
from struct_env.gym_sa_struct import GymSaStruct
from struct_env.struct_env import Struct

if __name__ == '__main__':
    print("test Struct")
    env=Struct()
    action_space = gym.spaces.Discrete(env.actions_per_agent)
    env.reset()
    done = False
    id =0
    while not done:
        action = {"agent_0":0,#action_space.sample(),
                  "agent_1":0}#action_space.sample()}
        observations, rewards, done = env.step(action)
        print(id, 'action:', action, 'reward:', rewards, done)
        id += 1
        done = done
    print()
    print()
    print("test GYM SA")
    env_sa = GymSaStruct()
    env_sa.reset()
    done = False
    id =0
    while not done:
        action = env_sa.action_space.sample()
        observation, reward, done, info = env_sa.step(action)
        print(id, 'action: ', action, 'reward:', reward, done)
        id += 1