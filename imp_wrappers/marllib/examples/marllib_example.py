from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY

from imp_wrappers.marllib.marllib_wrap_ma_struct import MarllibImpMarl
# Performed with a fresh conda environment and some downgrades
# conda env remove --name marllib
# conda create -n marllib python=3.8
# conda activate marllib
# pip install marllib
# pip install protobuf==3.20.1
# pip install gym==0.21.0


if __name__ == '__main__':
    # register new env
    ENV_REGISTRY["magym"] = MarllibImpMarl
    # initialize env
    env = marl.make_env(environment_name="imp-marl", map_name="all_scenario")
    # # pick mappo algorithms
    # mappo = marl.algos.mappo(hyperparam_source="test")
    # # customize model
    # model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
    # # start learning
    # mappo.fit(env, model, stop={'episode_reward_mean': 0, 'timesteps_total': 1000}, local_mode=True, num_gpus=0,
    #           num_workers=2, share_policy='all', checkpoint_freq=50)