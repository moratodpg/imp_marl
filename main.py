import itertools

import gym
from gym import spaces
from env.gym_sa_struct import GymSaStruct
from env.struct_env import Struct

if __name__ == '__main__':
    print("test")
    env=Struct()
    action_space = gym.spaces.Discrete(3)
    obs=env.reset()
    print(obs)
    print()
    print(env.step({"agent_0":action_space.sample(), "agent_1":action_space.sample()}))

    env_sa = GymSaStruct()
    print("reset SA",env_sa.reset())
    print(env_sa.action_space.sample())
    #print(env.step(env_sa.action_space.sample()))

