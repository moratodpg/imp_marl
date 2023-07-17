import os

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

from imp_wrappers.marllib.marllib_wrap_ma_struct import MarllibImpMarl

# Performed with a new conda environment and by installing MARLLIB from sources
# See https://github.com/Replicable-MARL/MARLlib

if __name__ == '__main__':
    # register new env
    ENV_REGISTRY["impmarl"] = MarllibImpMarl
    # initialize env
    config_path = os.path.join(os.path.dirname(__file__),
                               "config/impmarl.yaml")
    env = marl.make_env(environment_name="imp-marl", map_name="all_scenario",
                        abs_path=config_path)

    algo = marl.algos.mappo(hyperparam_source="common")
    # customize model
    model = marl.build_model(env, algo, {"core_arch": "gru"})
    # start learning
    algo.fit(env, model,
             stop={'episode_reward_mean': -15, 'timesteps_total': 100000},
             local_mode=True, num_gpus=0,
             num_workers=4, share_policy='all', checkpoint_freq=50)
