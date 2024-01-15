import datetime
import os
import pprint
import sys
import threading
import time
from copy import deepcopy
from os.path import abspath, dirname
from types import SimpleNamespace as SN

import numpy as np
import torch as th
import yaml
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY

from learners import REGISTRY as le_REGISTRY

from pymarl_train import _get_config, recursive_dict_update
from run import args_sanity_check
from runners import REGISTRY as r_REGISTRY
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, RunObserver
from sacred.utils import apply_backspaces_and_linefeeds
from utils.logging import get_logger, Logger

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
SETTINGS[
    "CAPTURE_MODE"
] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(abspath(__file__)), "results_test")


@ex.main
def my_main(_run, _config, _log, env_args):
    # Setting the random seed throughout the modules

    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    env_args["seed"] = _config["seed"]

    # run the framework
    run_test(_run, _config, _log)


def run_test(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = args.unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(results_path, "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential_test(args=args, logger=logger)

    time.sleep(30)  # To let sacred fileobserver write everything

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def run_sequential_test(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()

    args.n_agents = env_info["n_agents"]

    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["unit_dim"]
    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    groups = {"agents": args.n_agents}

    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    print("args.checkpoint_path ", args.checkpoint_path)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directory {} doesn't exist".format(args.checkpoint_path)
            )
            return
        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))
        timesteps = sorted(timesteps)
        print("timesteps", timesteps)
        # max_index = len(timesteps) - 1
        for idx_, timestep_to_load in enumerate(timesteps):
            # if idx_ < max_index:
            #     continue
            # if args.n_skip != 0 and idx_ % args.n_skip != 0:
            #     continue
            print("timestep_to_load", timestep_to_load)
            model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

            logger.console_logger.info("Loading model from {}".format(model_path))
            learner.load_models(model_path)
            runner.load_models(model_path)
            runner.t_env = timestep_to_load

            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            # new_big_buffer = ReplayBuffer(scheme, groups, n_test_runs * runner.batch_size,
            #               env_info["episode_limit"] + 1,
            #               preprocess=preprocess,
            #               device="cpu" if args.buffer_cpu_only else args.device)
            for i in range(n_test_runs):
                # Run for a whole episode at a time
                episode_batch = runner.run(test_mode=True)
                # new_big_buffer.insert_episode_batch(episode_batch)
                while episode_batch is None:
                    print("RESET")
                    episode_batch = runner.run(test_mode=True)
                    # new_big_buffer.insert_episode_batch(episode_batch)
            # runner.save_replay()
            # episode_sample = new_big_buffer.sample((n_test_runs) * runner.batch_size)
            # learner.stats(episode_sample, timestep_to_load)

        runner.close_env()
        logger.console_logger.info("Finished testing")

    else:
        logger.console_logger.info("Checkpoint directory doesn't exist")


def check_for_name(params):
    for param in params:
        if param.startswith("name="):
            return param[5:]
    return None


class SetID(RunObserver):
    priority = 50  # very high priority

    def __init__(self, custom_id):
        self.custom_id = custom_id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        return self.custom_id  # started_event should returns the _id


if __name__ == "__main__":
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")

    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    name = check_for_name(params)
    if name is not None:
        config_dict["name"] = name
    config_dict["unique_token"] = "{}__{}".format(
        config_dict["name"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(SetID(config_dict["unique_token"]))
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
