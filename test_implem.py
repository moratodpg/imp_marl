import itertools

import gym
from gym import spaces
from struct_env.gym_sa_struct import GymSaStruct
from struct_env.struct_env import Struct

if __name__ == '__main__':
    print("test")
    env=Struct()
    action_space = gym.spaces.Discrete(env.actions_per_agent)
    env.reset()
    done = False
    id =0
    while not done:
        action = {"agent_0":0,#action_space.sample(),
                  "agent_1":0}#action_space.sample()}
        observations, rewards, dones = env.step(action)
        print(id, 'act', action, 'r', rewards, dones)
        id += 1
        done = dones["__all__"]
    # env_sa = GymSaStruct()
    # print("reset SA",env_sa.reset())
    # print(env_sa.step(env_sa.action_space.sample()))

