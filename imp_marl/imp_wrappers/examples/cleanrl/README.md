# CleanRL

Clean RL is not a modular library but focus on a single-file implementation of RL aglos (see their [README](https://github.com/vwxyzjn/cleanrl/blob/master/README.md)).

Right now they do not support Gymnasium but they support old Gym (v21) that is implemented in the wrapper [**GymSaStruct**](imp_wrappers/gym/gym_wrap_sa_struct.py).

To use CleanRL:
1. [Install it](https://docs.cleanrl.dev/get-started/installation/).
2. Install imp_marl packages in your CleanRL python environment.
3. Register imp_marl as an available gym environments at the begining of CleanRL scripts. 
    ```
    from imp_marl.imp_wrappers.gym import gym_wrap_sa_struct
    gym.register("imp_marl", entry_point=gym_wrap_sa_struct.GymSaStruct)
    ```