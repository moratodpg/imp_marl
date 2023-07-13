""" Example of the use of the wrappers for a very basic usage of RLlib. """
import os

import ray
# Example coded with ray==2.5.1
from ray import air, tune
from ray.rllib.utils import try_import_tf, try_import_torch
from ray.tune import register_env
from ray.tune.registry import get_trainable_cls

from imp_wrappers.gymnasium.gym_wrap_sa_struct import GymSaStruct

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def env_gymsastruct_creator(env_config):
    return GymSaStruct(struct_type=env_config.get("struct_type", "struct"),
                       n_comp=env_config.get("n_comp", 2),
                       custom_param=env_config.get("custom_param", None),
                       discount_reward=env_config.get("discount_reward", .95),
                       state_obs=env_config.get("state_obs", True),
                       state_d_rate=env_config.get("state_d_rate", False),
                       state_alphas=env_config.get("state_alphas", False),
                       obs_d_rate=env_config.get("obs_d_rate", False),
                       obs_multiple=env_config.get("obs_multiple", False),
                       obs_all_d_rate=env_config.get("obs_all_d_rate", False),
                       obs_alphas=env_config.get("obs_alphas", False),
                       env_correlation=env_config.get("env_correlation",
                                                      False),
                       campaign_cost=env_config.get("campaign_cost", False))


if __name__ == '__main__':
    env_dict_config = {
        "struct_type": "struct",
        "n_comp": 2,
        "custom_param": None,
        "discount_reward": .95,
        "state_obs": True,
        "state_d_rate": False,
        "state_alphas": False,
        "obs_d_rate": False,
        "obs_multiple": False,
        "obs_all_d_rate": False,
        "obs_alphas": False,
        "env_correlation": False,
        "campaign_cost": False
    }

    ray.init()

    register_env("GymSaStruct", env_gymsastruct_creator)

    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        .environment("GymSaStruct", env_config=env_dict_config)
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    stop_iters = 100
    stop_timesteps = 1000
    stop_reward = 0

    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        "episode_reward_mean": stop_reward,
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop),
    )
    results = tuner.fit()

    ray.shutdown()
